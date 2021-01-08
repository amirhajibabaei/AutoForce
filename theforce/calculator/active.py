# +
from theforce.regression.gppotential import PosteriorPotential, PosteriorPotentialFromFolder
from theforce.descriptor.atoms import TorchAtoms, AtomsData, LocalsData
from theforce.similarity.sesoap import SeSoapKernel
from theforce.math.sesoap import SpecialRadii
from theforce.util.tensors import padded
from theforce.util.util import date, timestamp
from theforce.io.sgprio import SgprIO
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import Filter
import ase
from torch.autograd import grad
import torch
import numpy as np
import warnings
import os


def default_kernel(cutoff=6.):
    return SeSoapKernel(3, 3, 4, cutoff, radii=SpecialRadii({1: 0.5}))


class FilterDeltas(Filter):

    def __init__(self, atoms, shrink=0.95):
        """
        wraps atoms and causes a smooth variation of
        forces/stressed upon updating the ML potential
        """
        super().__init__(atoms, indices=[a.index for a in atoms])
        self.shrink = shrink
        self.f = 0
        self.s = 0

    def get_forces(self, *args, **kwargs):
        f = self.atoms.get_forces(*args, **kwargs)
        deltas = self.atoms.calc.deltas
        if deltas:
            self.f += deltas['forces']
        self.f *= self.shrink
        return f - self.f

    def get_stress(self, *args, **kwargs):
        s = self.atoms.get_stress(*args, **kwargs)
        deltas = self.atoms.calc.deltas
        if deltas:
            self.s += deltas['stress']
        self.s *= self.shrink
        return s - self.s

    def __getattr__(self, attr):
        return getattr(self.atoms, attr)


class ActiveCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, covariance=None, calculator=None, process_group=None,
                 ediff=0.041, fdiff=0.082, coveps=1e-4, covdiff=1e-2, meta=None,
                 logfile='active.log', tape='model.sgpr', **kwargs):
        """
        covariance:      None | similarity kernel(s) | path to a saved model | model
        calculator:      None | any ASE calculator
        process_group:   None | group
        ediff:           energy sensitivity
        fdiff:           forces sensitivity
        coveps:          covariance-loss ~ 0 if less than this value
        covdiff:         covariance-loss sensitivity heuristic
        meta:            meta energy calculator
        logfile:         string | None
        tape:            string (with suffix .sgpr), the file used to save/load updates
        kwargs:          ASE's calculator kwargs

        *** important ***
        All of the updates are recorded in the tape.
        If the tape already exists, it will be loaded.

        *** important ***
        You may wants to wrap atoms with FilterDeltas if you intend to 
        carry out molecular dynamics simulations. 

        *** important ***
        You can use log_to_figure function in this module for visualization.
            e.g. log_to_figure('active.log')

        --------------------------------------------------------------------------------------
        Notes:

        If covariance is None, the default kernel will be used.
        At the beginning, covariance is often a list of similarity kernels:
            e.g. theforce.similarity.sesoap.SeSoapKernel(...)
        Later we can use an existing model.
        A trained model can be saved with:
            e.g. calc.model.to_folder('model/')
        An existing model is loaded with:
            e.g. ActiveCalculator('model/', ...)

        In case one wishes to use an existing model without further updates, 
        then pass "calculator=None".

        For parallelism, import "mpi_init" from theforce.util.parallel,
        then set 
            process_group = mpi_init()
        as kwarg when creating the ActiveCalculator.

        Setting a finite covdiff (0.01~0.1) may be necessary if the training
        starts with an empty model and a state with zero initial forces.
        If covariance-loss < coveps, the update trial will be skipped.
        Depending on the problem at hand, coveps can be chosen as high as
        1e-2 which can speed up the calculator by potentially a few orders
        of magnitude.
        If the model is volatile, one can set coveps=0 for robustness.

        The tape arg is the name of the file used for saving the 
        updates (the added data and inducing). 
        The model can be regenarated using this file.
        "tape" files are never overwritten (allways appended).
        This file can also be used to import other tapes by a line 
        containing 
            include: path-to-another-tape
        """
        Calculator.__init__(self, **kwargs)
        self._calc = calculator
        self.process_group = process_group
        self.get_model(covariance or default_kernel())
        self.ediff = ediff
        self.fdiff = fdiff
        self.coveps = coveps
        self.covdiff = covdiff
        self.meta = meta
        self.logfile = logfile
        self.stdout = True
        self.step = 0
        self.log('active calculator says Hello!', mode='w')
        self.log_settings()
        self.log('model size: {} {}'.format(*self.size))
        self.tape = SgprIO(tape)
        read_tape = not (type(covariance) == str and not self.active)
        if os.path.isfile(tape) and read_tape:
            call, cadd = self.model.include(self.tape)
            self.log(f'{tape}: {call} -> {cadd}')
        if self.active:
            self.tape.write_params(ediff=self.ediff, fdiff=self.fdiff)
        self.normalized = None

    @property
    def active(self):
        return self._calc is not None

    def get_model(self, model):
        if type(model) == str:
            self.model = PosteriorPotentialFromFolder(
                model, load_data=True, update_data=self.active, group=self.process_group)
        elif type(model) == PosteriorPotential:
            self.model = model
        else:
            self.model = PosteriorPotential(model)

    @property
    def size(self):
        return self.model.ndata, len(self.model.X)

    def calculate(self, _atoms=None, properties=['energy'], system_changes=all_changes):

        if type(_atoms) == ase.atoms.Atoms:
            atoms = TorchAtoms(ase_atoms=_atoms)
            uargs = {'cutoff': self.model.cutoff,
                     'descriptors': self.model.gp.kern.kernels}
            self.to_ase = True
        else:
            atoms = _atoms
            uargs = {}
            self.to_ase = False
        if _atoms is not None and self.process_group is not None:
            atoms.attach_process_group(self.process_group)
        Calculator.calculate(self, atoms, properties, system_changes)
        self.atoms.update(posgrad=True, cellgrad=True,
                          forced=True, dont_save_grads=True, **uargs)

        # build a model
        data = True
        if self.step == 0:
            if self.active and self.model.ndata == 0:
                self.initiate_model()
                data = False

        # kernel
        self.cov = self.model.gp.kern(self.atoms, self.model.X)

        # energy/forces
        energies = self.cov@self.model.mu
        retain_graph = self.active or (self.meta is not None)
        energy = self.reduce(energies, retain_graph=retain_graph)

        # ... or
        # self.allcov = self.gather(self.cov)
        # energies = self.allcov@self.model.mu
        # retain_graph = self.active or (self.meta is not None)
        # energy = self.reduce(energies, retain_graph=retain_graph, reduced=True)

        # active learning
        self.deltas = None
        self.covlog = ''
        if self.active:
            m, n = self.update(data=data)
            if n > 0 or m > 0:  # update results
                pre = self.results.copy()
                energies = self.cov@self.model.mu
                retain_graph = self.meta is not None
                energy = self.reduce(energies, retain_graph=retain_graph)
                self.deltas = {}
                for quant in ['energy', 'forces', 'stress']:
                    self.deltas[quant] = self.results[quant] - pre[quant]

        # meta terms
        meta = ''
        if self.meta is not None:
            energies, kwargs = self.meta(self)
            if energies is not None:
                meta_energy = self.reduce(energies, **kwargs)
                meta = f'meta: {meta_energy}'

        # step
        self.log('{} {} {} {}'.format(energy, self.atoms.get_temperature(),
                                      self.covlog, meta))
        self.step += 1

        # needed for self.calculate_numerical_stress
        self.results['free_energy'] = self.results['energy']

    def reduce(self, local_energies, op='=', retain_graph=False, reduced=False):
        energy = local_energies.sum()
        if self.atoms.is_distributed and not reduced:
            torch.distributed.all_reduce(energy)
        forces, stress = self.grads(energy, retain_graph=retain_graph)
        if op == '=':
            self.results['energy'] = energy.detach().numpy()
            self.results['forces'] = forces.detach().numpy()
            self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
        elif op == '+=':
            self.results['energy'] += energy.detach().numpy()
            self.results['forces'] += forces.detach().numpy()
            self.results['stress'] += stress.flat[[0, 4, 8, 5, 2, 1]]
        return float(energy)

    def zero(self):
        self.results['energy'] = 0.
        self.results['forces'] = 0.
        self.results['stress'] = 0.

    def grads(self, energy, retain_graph=False):
        if energy.grad_fn:
            forces = -grad(energy, self.atoms.xyz, retain_graph=True,
                           allow_unused=True)[0]
            cellgrad, = grad(energy, self.atoms.lll, retain_graph=retain_graph,
                             allow_unused=True)
            if cellgrad is None:
                cellgrad = torch.zeros_like(self.atoms.lll)
        else:
            forces = torch.zeros_like(self.atoms.xyz)
            cellgrad = torch.zeros_like(self.atoms.lll)
        if self.atoms.is_distributed:
            torch.distributed.all_reduce(forces)
            torch.distributed.all_reduce(cellgrad)
        # stress
        stress1 = -(forces[:, None]*self.atoms.xyz[..., None]).sum(dim=0)
        stress2 = (cellgrad[:, None]*self.atoms.lll[..., None]).sum(dim=0)
        try:
            volume = self.atoms.get_volume()
        except ValueError:
            volume = -2  # here stress2=0, thus trace(stress) = virial (?)
        stress = (stress1 + stress2).detach().numpy() / volume
        return forces, stress

    def initiate_model(self):
        data = AtomsData([self.snapshot()])
        i = self.atoms.first_of_each_atom_type()
        inducing = LocalsData([self.atoms.local(j, detach=True) for j in i])
        self.model.set_data(data, inducing)
        # data is stored in _exact, thus we only store the inducing
        for loc in inducing:
            self.tape.write(loc)
        details = [(j, self.atoms.numbers[j]) for j in i]
        self.log('seed size: {} {} details: {}'.format(
            *self.size, details))
        self.optimize()

    def _exact(self, copy):
        tmp = copy.as_ase() if self.to_ase else copy
        tmp.set_calculator(self._calc)
        energy = tmp.get_potential_energy()
        forces = tmp.get_forces()
        self.tape.write(tmp)
        self.log('exact energy: {}'.format(energy))
        #
        if self.model.ndata > 0:
            dE = self.results['energy'] - energy
            df = abs(self.results['forces'] - forces)
            self.log('errors (pre):  del-E: {:.2g}  max|del-F|: {:.2g}  mean|del-F|: {:.2g}'.format(
                dE, df.max(), df.mean()))
        return energy, forces

    def snapshot(self, fake=False, copy=None):
        if copy is None:
            copy = self.atoms.copy()
        if fake:
            energy = self.results['energy']
            forces = self.results['forces']
        else:
            energy, forces = self._exact(copy)
        copy.set_calculator(SinglePointCalculator(copy, energy=energy,
                                                  forces=forces))
        copy.set_targets()
        return copy

    def head(self, energy_and_forces=None):
        added = self.model.data[-1]
        if energy_and_forces is None:
            energy, forces = self._exact(added)
        added.calc.results['energy'] = energy
        added.calc.results['forces'] = forces
        added.set_targets()
        self.model.make_munu()

    def scatter(self, x, dim=0):
        if self.atoms.is_distributed:
            index = torch.tensor(self.atoms.indices)
            return x.index_select(dim, index)
        else:
            return x

    def gather(self, x):
        if self.atoms.is_distributed:
            size = [s for s in x.size()]
            size[0] = self.atoms.natoms
            _x = torch.zeros(*size)
            _x[self.atoms.indices] = x
            torch.distributed.all_reduce(_x)
            return _x
        else:
            return x

    def get_covloss(self):
        b = self.model.choli@self.cov.detach().t()
        c = (b*b).sum(dim=0)
        if not self.normalized:
            alpha = [self.model.gp.kern(x, x).detach()
                     for x in self.atoms]
            alpha.append(torch.zeros(0))
            alpha = torch.cat(alpha).view(-1)
            c = c/alpha
            if self.normalized is None:
                self.normalized = self.gather(alpha).allclose(torch.ones([]))
                self.log(f'kernel normalization status {self.normalized}')
        if c.size(0) > 0:
            beta = (1 - c).clamp(min=0.).sqrt()
        else:
            beta = c
        beta = self.gather(beta)
        return beta

    def update_inducing(self):
        added_beta = 0
        added_diff = 0
        added_indices = []
        added_covloss = None
        self.blind = False
        while True:
            if len(added_indices) == self.atoms.natoms:
                break
            beta = self.get_covloss()
            q = torch.argsort(beta, descending=True)
            if beta[q[0]] < self.coveps:
                break
            for k in q.tolist():
                if k not in added_indices:
                    break
            if beta[k].isclose(torch.ones([])):
                self.blind = True
            loc = self.atoms.local(k, detach=True)
            if loc.number in self.model.gp.species:
                if beta[k] > self.covdiff and self.model.indu_counts[loc.number] < 2:
                    self.model.add_inducing(loc)
                    self.tape.write(loc)
                    added_beta += 1
                    x = self.model.gp.kern(self.atoms, loc)
                    self.cov = torch.cat([self.cov, x], dim=1)
                    added_indices.append(k)
                    added_covloss = beta[k]
                    self.blind = True
                    self.optimize()
                else:
                    _ediff = (self.ediff if len(self.model.X) > 1
                              else torch.finfo().eps)
                    added, delta = self.model.add_1inducing(
                        loc, _ediff, detach=False)
                    self.log_cov(beta[k], delta)
                    if added:
                        self.tape.write(loc)
                        added_diff += 1
                        x = self.model.gp.kern(self.atoms, loc)
                        self.cov = torch.cat([self.cov, x], dim=1)
                        added_indices.append(k)
                        added_covloss = beta[k]
                        self.optimize()
                    else:
                        break
        added = added_beta + added_diff
        if added > 0:
            details = [(k, self.atoms.numbers[k]) for k in added_indices]
            self.log('added indu: {} ({},{}) -> size: {} {} details: {:.2g} {}'.format(
                added, added_beta, added_diff, *self.size, added_covloss, details))
            if not self.normalized:
                self.log(f'kernel diag mean: {self.model.kern_diag_mean}')
            if self.blind:
                self.log('model may be blind -> go robust')
        self.covlog = f'{float(beta[q[0]])}'
        return added

    def update_data(self, try_fake=True):
        n = self.model.ndata
        new = self.snapshot(fake=try_fake)
        self.model.add_1atoms(new, self.ediff, self.fdiff)
        added = self.model.ndata - n
        if added > 0:
            if try_fake:
                self.head()
            self.log('added data: {} -> size: {} {}'.format(
                added, *self.size))
            self.optimize()
        return added

    def optimize(self):
        self.model.optimize_model_parameters(ediff=self.ediff, fdiff=self.fdiff)

    def update(self, inducing=True, data=True):
        m = self.update_inducing() if inducing else 0
        n = self.update_data(try_fake=not self.blind) if m > 0 and data else 0
        if m > 0 or n > 0:
            self.log('fit error (mean,std): E: {:.2g} {:.2g}   F: {:.2g} {:.2g}   R2: {:.4g}'.format(
                *(float(v) for v in self.model._stats)))
            if self.rank == 0:
                self.log(f'noise: {self.model.scaled_noise}')
        # tunning noise is unstable!
        # if n > 0 and not self.model.is_well():
        #    self.log(f'tuning noise: {self.model.gp.noise.signal} ->')
        #    self.model.tune_noise(min_steps=10, verbose=self.logfile)
        #    self.log(f'noise: {self.model.gp.noise.signal}')
        return m, n

    @property
    def rank(self):
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        else:
            return 0

    def log(self, mssge, mode='a'):
        if self.logfile and self.rank == 0:
            with open(self.logfile, mode) as f:
                f.write('{} {} {}\n'.format(date(), self.step, mssge))
                if self.stdout:
                    print('{} {} {}'.format(date(), self.step, mssge))
            # cov log
            if mode == 'w' and False:
                with open('cov.log', mode) as f:
                    f.write('# covariance data\n')

    def log_cov(self, *args):
        if self.logfile and self.rank == 0 and False:
            with open('cov.log', 'a') as f:
                f.write(' '.join([str(float(arg)) for arg in args])+'\n')

    def log_settings(self):
        settings = ['ediff', 'fdiff', 'coveps', 'covdiff']
        s = ''.join([f' {s}: {getattr(self, s)} ' for s in settings])
        self.log(f'settings: {s}')


class Meta:

    def __init__(self, scale=1e-2):
        self.scale = scale
        self.pot = None

    def __call__(self, calc):
        if self.pot is None:
            self.pot = torch.zeros(calc.cov.size(1))
        cov = calc.gather(calc.cov)
        nu = calc.model.Mi@cov.t()
        norm = (cov@nu).sum().sqrt()
        mu = nu.detach().sum(dim=1)/norm.detach()
        self.pot = padded(self.pot, mu.size()) + self.scale*mu
        energies = (cov@self.pot).sum()/norm
        kwargs = {'op': '+=', 'reduced': True}
        return energies, kwargs


def parse_logfile(file='active.log', window=(None, None)):
    start = None
    settings = []
    elapsed = []
    energies = []
    temperatures = []
    covloss = []
    exact_energies = []
    indu = []
    errors = []
    fit = []
    meta = []
    for line in open(file):
        s = line.split()
        ts = timestamp(' '.join(s[:2]))
        if start is None:
            start = ts
        ts = (ts-start)/60
        split = s[2:]

        if split[1] == 'settings:':
            settings = {a: float(b) for a, b in zip(split[2::2], split[3::2])}

        try:
            step = int(split[0])
            if window[0] and step < window[0]:
                continue
            if window[1] and step > window[1]:
                break
        except IndexError:
            continue

        elapsed += [(step, ts)]

        try:
            energies += [(step, float(split[1]))]
            temperatures += [(step, float(split[2]))]
            covloss += [(step, float(split[3]))]
            if 'meta:' in split:
                meta += [(step, float(split[5]))]
        except:
            pass

        if 'exact energy' in line:
            exact_energies += [(step, float(split[3]))]

        if 'added indu' in line:
            sf = float(split[split.index('details:') + 1])
            indu += [(step, sf)]

        if 'errors' in line:
            errors += [(step, [float(v) for v in split[4:8:2]])]

        if 'fit' in line:
            fit += [(step, [float(split[k]) for k in [-7, -6, -4, -3, -1]])]
    return energies, exact_energies, temperatures, covloss, meta, indu, fit, elapsed, settings


def log_to_figure(file, figsize=(10, 5), window=(None, None), meta_ax=True):
    import pylab as plt
    ml, fp, tem, covloss, meta, indu, fit, elapsed, settings = parse_logfile(
        file, window=window)
    fig, _axes = plt.subplots(2, 2, figsize=figsize)
    axes = _axes.reshape(-1)
    # 0
    x, y = zip(*ml)
    axes[0].plot(x, y, label='ML', zorder=1)
    if len(fp) > 0:
        r, s = zip(*fp)
        axes[0].scatter(r, s, color='r', label='ab initio', zorder=2)
    axes[0].set_ylabel('potential')
    axes[0].legend()
    if len(meta) > 0 and meta_ax:
        ax_meta = axes[0].twinx()
        ax_meta.plot(*zip(*meta), color='goldenrod', lw=0.5)
        ax_meta.set_ylabel('meta')
    # 1
    axes[1].plot(*zip(*tem))
    axes[1].set_ylabel('temperature')
    # 2
    axes[2].plot(*zip(*covloss), label='max', zorder=1)
    axes[2].set_ylabel('cov-loss')
    axes[2].set_ylim(1e-5, 1e-1)
    axes[2].set_yscale('log')
    if len(indu) > 0:
        axes[2].scatter(*zip(*indu), color='lime', label='added', zorder=2)
    axes[2].legend()
    wall = axes[2].twinx()
    wall.plot(*zip(*elapsed), color='cyan', alpha=0.5)
    wall.set_ylabel('minutes')
    axes[2].axhline(y=settings['coveps:'], ls='--', color='k')
    axes[2].axhline(y=settings['covdiff:'], ls='--', color='k', alpha=0.3)
    axes[2].grid()
    # 3
    if len(fit) > 0:
        p, q = zip(*fit)
        q = np.array(q)
        q[:, 0:2] *= 10
        axes[3].axhline(y=0, ls=':', lw=1, color='k')
        axes[3].fill_between(p, q[:, 2]-q[:, 3], q[:, 2] + q[:, 3], color='cornflowerblue',
                             interpolate=True, alpha=0.5, label=r'$\Delta f$')
        axes[3].scatter(p, q[:, 2], color='cornflowerblue')
        axes[3].fill_between(p, q[:, 0]-q[:, 1], q[:, 0] + q[:, 1], color='salmon',
                             interpolate=True, alpha=0.5, label=r'$10\times\Delta(E/N)$')
        axes[3].scatter(p, q[:, 0], color='salmon')
        axes[3].grid()
        axes[3].legend()
        axes[3].set_ylabel('Errors')
        # R2
        if q.shape[1] > 4:
            R2 = axes[3].twinx()
            R2.plot(p, 1-q[:, 4], ':', color='grey')
            R2.set_ylabel(r'$1-R^2$')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    import sys
    log = sys.argv[1]
    fig = log_to_figure(log)
    fig.savefig(log.replace('.log', '.pdf'))
