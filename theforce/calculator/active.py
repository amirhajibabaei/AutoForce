# +
from theforce.regression.gppotential import PosteriorPotential, PosteriorPotentialFromFolder
from theforce.descriptor.atoms import TorchAtoms, AtomsData, LocalsData
from theforce.util.tensors import padded
from theforce.util.util import date
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import Filter
import ase
from torch.autograd import grad
import torch
import numpy as np
import warnings


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

    def get_forces(self):
        f = self.atoms.get_forces()
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

    def __init__(self, covariance, calculator=None, process_group=None, ediff=0.1, fdiff=0.1, covdiff=0.1,
                 coveps=1e-4, meta=None, logfile='active.log', storage='storage.traj', **kwargs):
        """
        covariance:      similarity kernel(s) | path to a saved model | model
        calculator:      None | any ASE calculator
        process_group:   None | group
        ediff:           energy sensitivity
        fdiff:           forces sensitivity
        covdiff:         covariance-loss sensitivity heuristic
        coveps:          covariance-loss ~ 0 if less than this value
        meta:            meta energy calculator
        logfile:         string | None
        storage:         string | None
        kwargs:          ASE's calculator kwargs

        *** important ***
        wrap atoms with FilterDeltas if you intend to carry out
        molecular dynamics simulations.

        --------------------------------------------------------------------------------------

        At the beginning, covariance is often a list of similarity kernels:
            e.g. theforce.similarity.universal.UniversalSoapKernel(...)
        Later we can use an existing model.
        A trained model can be saved with:
            e.g. calc.model.to_folder('model/')
        An existing model is loaded with:
            e.g. ActiveCalculator('model/', ...)

        In case one wishes to use an existing model without further updates, 
        then pass "calculator=None".

        For parallelism, first call:
            torch.init_process_group('mpi')
        then, set process_group=torch.distributed.group.WORLD in kwargs.

        Sensitivity params can be changed on-the-fly:
            e.g. calc.ediff = 0.05

        If covariance-loss (range [0,1]) for a LCE is greater than covdiff,
        it will be automaticaly added to the inducing set.
        Moreover, exact calculations will be triggered.
        Do not make covdiff too small!
        covdiff=1 eliminates this heuristic, if one wishes to keep the
        algorithm non-parametric.
        Setting a finite covdiff (~0.1) may be necessary if the training
        starts from a state with zero forces (with an empty model).
        If covariance-loss < coveps, the update trial will be skipped.
        Depending on the problem at hand, coveps can be chosen as high as
        1e-2 which can speed up the calculator by potentially a few orders
        of magnitude.
        If the model is volatile, one can set coveps=0 for robustness.

        The "storage" arg is the name of the file used for saving the exact
        calculations (with the main calculator). Turn it of by "storage=None".
        """
        Calculator.__init__(self, **kwargs)
        self._calc = calculator
        self.process_group = process_group
        self.get_model(covariance)
        self.ediff = ediff
        self.fdiff = fdiff
        self.covdiff = covdiff
        self.coveps = coveps
        self.meta = meta
        self.logfile = logfile
        self.stdout = True
        self.storage = storage
        self.normalized = None
        self.step = 0

    @property
    def active(self):
        return self._calc is not None

    def get_model(self, model):
        if type(model) == str:
            self.model = PosteriorPotentialFromFolder(
                model, load_data=self.active, group=self.process_group)
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
            self.log('active calculator says Hello!', mode='w')
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
        details = [(j, self.atoms.numbers[j]) for j in i]
        self.log('seed size: {} {} details: {}'.format(
            *self.size, details))

    def _exact(self, copy):
        tmp = copy.as_ase() if self.to_ase else copy
        tmp.set_calculator(self._calc)
        energy = tmp.get_potential_energy()
        forces = tmp.get_forces()
        if self.storage and self.rank == 0:
            ase.io.Trajectory(self.storage, 'a').write(tmp)
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
            if q[0] < self.coveps:
                break
            for k in q.tolist():
                if k not in added_indices:
                    break
            if beta[k].isclose(torch.ones([])):
                self.blind = True
            loc = self.atoms.local(k, detach=True)
            if loc.number in self.model.gp.species:
                if beta[k] > self.covdiff:
                    self.model.add_inducing(loc)
                    added_beta += 1
                    x = self.model.gp.kern(self.atoms, loc)
                    self.cov = torch.cat([self.cov, x], dim=1)
                    added_indices.append(k)
                    added_covloss = beta[k]
                    self.blind = True
                else:
                    _ediff = (self.ediff if len(self.model.X) > 1
                              else torch.finfo().eps)
                    added, delta = self.model.add_1inducing(
                        loc, _ediff, detach=False)
                    if added:
                        added_diff += 1
                        x = self.model.gp.kern(self.atoms, loc)
                        self.cov = torch.cat([self.cov, x], dim=1)
                        added_indices.append(k)
                        added_covloss = beta[k]
                    else:
                        break
        added = added_beta + added_diff
        if added > 0:
            details = [(k, self.atoms.numbers[k]) for k in added_indices]
            self.log('added indu: {} ({},{}) -> size: {} {} details: {:.2g} {}'.format(
                added, added_beta, added_diff, *self.size, added_covloss, details))
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
        return added

    def update(self, inducing=True, data=True):
        m = self.update_inducing() if inducing else 0
        n = self.update_data(try_fake=not self.blind) if m > 0 and data else 0
        if m > 0 or n > 0:
            self.log('fit error (mean,std): E: {:.2g} {:.2g}   F: {:.2g} {:.2g}'.format(
                *(float(v) for v in self.model._stats)))
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


def parse_logfile(file='active.log'):
    energies = []
    temperatures = []
    covloss = []
    exact_energies = []
    indu = []
    errors = []
    fit = []
    meta = []
    for line in open(file):
        split = line.split()[2:]

        try:
            step = int(split[0])
        except IndexError:
            continue

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
            fit += [(step, [float(split[k]) for k in [-5, -4, -2, -1]])]
    return energies, exact_energies, temperatures, covloss, meta, indu, fit


def log_to_figure(file, figsize=(10, 5)):
    import pylab as plt
    ml, fp, tem, covloss, meta, indu, fit = parse_logfile(file)
    fig, _axes = plt.subplots(2, 2, figsize=figsize)
    axes = _axes.reshape(-1)
    # 0
    x, y = zip(*ml)
    axes[0].plot(x, y, label='ML')
    if len(fp) > 0:
        r, s = zip(*fp)
        axes[0].scatter(r, s, color='r', label='FP')
    axes[0].set_ylabel('potential')
    axes[0].legend()
    if len(meta) > 0:
        ax_meta = axes[0].twinx()
        ax_meta.plot(*zip(*meta), color='lime')
        ax_meta.set_ylabel('meta')
    # 1
    axes[1].plot(*zip(*tem))
    axes[1].set_ylabel('temperature')
    # 2
    axes[2].plot(*zip(*covloss), label='max')
    axes[2].set_ylabel('cov-loss')
    axes[2].set_ylim(1e-4, 0.1)
    axes[2].set_yscale('log')
    if len(indu) > 0:
        axes[2].scatter(*zip(*indu), color='lime', label='added')
    axes[2].legend()
    # 3
    if len(fit) > 0:
        p, q = zip(*fit)
        q = np.array(q)
        axes[3].errorbar(p, q[:, 0], yerr=q[:, 1], color='r', capsize=5)
        ax_ferr = axes[3].twinx()
        ax_ferr.errorbar(p, q[:, 2], yerr=q[:, 3],
                         color='b', capsize=5, alpha=0.5)
        axes[3].set_ylabel('E-err')
        ax_ferr.set_ylabel('F-err')
    fig.tight_layout()
    return fig
