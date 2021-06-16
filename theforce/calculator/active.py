# +
from theforce.regression.gppotential import PosteriorPotential, PosteriorPotentialFromFolder
from theforce.descriptor.atoms import TorchAtoms, AtomsData, LocalsData
from theforce.similarity.sesoap import SeSoapKernel, SubSeSoapKernel
from theforce.descriptor.sesoap import DefaultRadii
from theforce.util.tensors import padded, nan_to_num
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


def default_kernel(lmax=3, nmax=3, exponent=4, cutoff=6., species=None):
    if species is None:
        kern = SeSoapKernel(lmax, nmax, exponent, cutoff, radii=DefaultRadii())
    else:
        kern = [SubSeSoapKernel(lmax, nmax, exponent, cutoff, z, species, radii=DefaultRadii())
                for z in species]
    return kern


def clamp_forces(f, m):
    g = np.where(f > m, m, f)
    h = np.where(g < -m, -m, g)
    return h


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
        g = clamp_forces(self.f, 1.)
        return f - g

    def get_stress(self, *args, **kwargs):
        s = self.atoms.get_stress(*args, **kwargs)
        deltas = self.atoms.calc.deltas
        if deltas:
            self.s += deltas['stress']
        self.s *= self.shrink
        return s - self.s

    def __getattr__(self, attr):
        return getattr(self.atoms, attr)


kcal_mol = 0.043
inf = float('inf')


class ActiveCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, covariance='pckl', calculator=None, process_group=None, meta=None,
                 logfile='active.log', pckl='model.pckl', tape='model.sgpr', test=None, stdout=False,
                 ediff=2*kcal_mol, ediff_lb=None, ediff_ub=None,
                 ediff_tot=4*kcal_mol, fdiff=3*kcal_mol, noise_f=kcal_mol, ioptim=1,
                 max_data=inf, max_inducing=inf, kernel_kw=None, veto=None, include_params=None):
        """
        inputs:
            covariance:      None | similarity kernel(s) | path to a pickled model | model
            calculator:      None | any ASE calculator
            process_group:   None | group
            meta:            meta energy calculator

        outputs:
            logfile:         string | None
            pckl:            string | None, folder for pickling the model
            tape:            string (with suffix .sgpr), the file used for saving updates
            test:            None | integer for independent testing intervals
            stdout:          bool, if True prints the log also to stdout

        optimization and sampling:
            ediff:           local energy sensitivity (eV)
            ediff_lb:        lower-bound for ediff | None -> ediff
            ediff_ub:        upper-bound for ediff | None -> ediff
            ediff_tot:       total energy sensitivity (eV) | inf is allowed
            fdiff:           forces sensitivity (eV/A) | inf is allowed
            noise_f:         bias force (fit) MAE to this value
            max_data:        if data exceeds this limit, remove earlier data
            max_inducing     if inducing exceeds this limit, remove earlier inducing

        control:
            kernel_kw:       kwargs passed when default_kernel is called
            ioptim:          a tag for hyper-parameter optimization
            veto:            dict, for vetoing ML updates e.g. {'forces': 5.}
            include_params:  used when include_data, include_tape is called

        callables:
            include_data     for modeling the existing data
            include_tape     for training from a sgpr tape

        *** important ***
        You may wants to wrap atoms with FilterDeltas if you intend to
        carry out molecular dynamics simulations.

        *** important ***
        For training the model with existing data use the following
            calc.include_data(data)
        where data can be a list of atoms with energy and forces already available
        or the path to a traj file.

        *** important ***
        For training a model with an existing sgpr file use
            calc.include_tape(tape)

        *** important ***
        You can use log_to_figure function in this module for visualization.
            e.g. log_to_figure('active.log')

        --------------------------------------------------------------------------------------
        Notes:

        covariance:
            At the beginning, covariance is often a list of similarity kernels:
                e.g. theforce.similarity.sesoap.SeSoapKernel(...)
            If covariance is None, the default kernel will be used.
            Later we can use an existing model. A trained model can be pickled with:
                e.g. calc.model.to_folder('model/')
            A pickled model is loaded with:
                e.g. ActiveCalculator('model/', ...)
            By default, the model will be automatically pickled after every update
            unless pckl=None.

        calculator:
            The ab initio calculator which the model intends to learn.
            In case one wishes to use an existing model without further updates,
            then pass "calculator=None".

        process_group:
            For parallelism, import "mpi_init" from theforce.util.parallel,
            then set
                process_group = mpi_init()
            as kwarg when creating the ActiveCalculator.

        pckl:
            The model will be pickled after every update in this folder
            which can be loaded in the future simulations simply by
                calc = ActiveCalculator(pckl)
            This way, there is no overhead for rebuilding the model from
            scratch.

        tape:
            The tape arg is the name of the file used for saving the
            updates (the added data and inducing LCEs).
            "tape" files are never overwritten (allways appended).
            These files can be used for rebuilding the model with different
            parameters, combining models, and in general post processing.
            A tape can be viewed as the important information along the
            trajectory. A tape can be used for training a model by
                calc.include_tape(file)

        test:
            For instance, if test=100 and 100 steps have passed since the last
            exact calculation, an exact calculation will be performed.
            This can be used for monitoring the on-the-fly ML accuracy and
            has no effect on training. The exact calculation (FP) will be saved
            in 'active_FP.traj' while the models predictions (ML) will be saved
            in 'active_ML.traj'. These files will be overwritten in the next
            simulation. The following command can be used for a quick description
            of ML errors

                python -m theforce.regression.scores active_ML.traj active_FP.traj

        ediff, ediff_lb, ediff_ub: -> for sampling the LCEs for sparse representation
            ediff controls the LCE sampling rate. You decrease ediff for higher accuracy
            or increase ediff for higher speed.
            The other two parameters are only experimental and have no effect if they
            are equal to ediff; which is currently the default.

        ediff_tot, fdiff: -> for samplig the ab initio data
            These are the thresholds for accepting a sample for ab initio calculations.
            If one of them is inf, it will be ignored. Thus you can set ediff_tot=inf
            for focusing on forces or vice versa.

        kernel_kw:
            If covariance = None, default kernel is used. kernel_kw can be used for
            passing some parameters for initiating the kernel. kernel_kw = None is
            equivalent to {'lmax': 3, 'nmax': 3, 'exponent': 4, 'cutoff': 6.}.
            e.g. kernel_kw = {'cutoff': 7.0} changes the cutoff to 7.

        ioptim:
            For setting the hyper-parameter optimization (HPO) frequency. In 1 MD step,
            potentially several LCEs can be sampled as the inducing/reference set.
            If at least 1 LCE is sampled, then a single-point ab initio calculation
            maybe performed and the configuration sampled as additional data for the
            regression. Depending on ioptim tag, HPO can be invoked with different
            frequencies

               -1 -> no HPO
                0 -> once for every LCE/data sampled
                1 -> only if n.o. LCE + n.o. data sampled > 0
                2 -> only if new data are sampled
            i > 2 -> only when n.o. sampled data is divisible by (i-1)

            Frequency of HPOs decrease dramatically with increasing ioptim:
                0 >> 1 >> 2
            Default is ioptim = 1.

        veto:
            e.g. {'forces': 5.} -> do not try ML update if forces > 5.
            This is usefull for algorithms such as random structure search where
            many high energy stuctures maybe generated. With vetoing ML updates
            for these structures the computational cost is reduced.

        include_params (deprecated in favor of veto):
            This applies some constraints for sampling data when include_data/-tape are called.
            e.g. {'fmax': 5.0}, etc.
        """

        Calculator.__init__(self)
        self._calc = calculator
        self.process_group = process_group
        self.pckl = pckl
        self.get_model(covariance, kernel_kw or {})
        self.ediff = ediff
        self.ediff_lb = ediff_lb or self.ediff
        self.ediff_ub = ediff_ub or self.ediff
        self.ediff_tot = ediff_tot
        self.fdiff = fdiff
        self.noise_f = noise_f
        self.ioptim = ioptim
        self._ioptim = 0
        self.max_data = max_data
        self.max_inducing = max_inducing
        self.meta = meta
        self.logfile = logfile
        self.stdout = stdout
        self.step = 0
        self.log('active calculator says Hello!', mode='w')
        self.log(f'kernel: {self.model.descriptors}')
        self.log_settings()
        self.log('model size: {} {}'.format(*self.size))
        self.tape = None if tape is None else SgprIO(tape)
        # if self.active and self.tape:
        #    self.tape.write_params(ediff=self.ediff, ediff_tot=self.ediff_tot,
        #                           fdiff=self.fdiff)
        self.test = test
        self._last_test = 0
        self._ktest = 0
        self.normalized = None
        self.updated = False
        self._update_args = {}
        self._veto = {} if veto is None else veto
        self.include_params = {'fmax': float('inf')}
        if include_params:
            self.include_params.update(include_params)
        self.tune_for_md = True

    @property
    def active(self):
        return self._calc is not None

    def get_model(self, model, kernel_kw):
        if model == 'pckl':
            if os.path.isdir(self.pckl):
                model = self.pckl
            else:
                model = None
        if model is None:
            model = default_kernel(**kernel_kw)
        self._ready = True
        if type(model) == str:
            self.model = PosteriorPotentialFromFolder(
                model, load_data=True, update_data=False, group=self.process_group)
            self._ready = False
        elif type(model) == PosteriorPotential:
            self.model = model
        else:
            self.model = PosteriorPotential(model)

    def get_ready(self):
        if not self._ready:
            self.model.data.update(cutoff=self.model.cutoff,
                                   descriptors=self.model.descriptors)
            self._ready = True

    @property
    def size(self):
        return self.model.ndata, len(self.model.X)

    def calculate(self, _atoms=None, properties=['energy'], system_changes=all_changes):

        if self.size[1] == 0 and not self.active:
            raise RuntimeError('you forgot to assign a DFT calculator!')

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
        if self.step == 0:
            if self.active and self.model.ndata == 0:
                self.initiate_model()
                self._update_args = dict(data=False)

        # kernel
        self.cov = self.model.gp.kern(self.atoms, self.model.X)

        # energy/forces
        self.update_results(self.active or (self.meta is not None))

        # active learning
        self.deltas = None
        self.covlog = ''
        if self.active and not self.veto():
            pre = self.results.copy()
            m, n = self.update(**self._update_args)
            if n > 0 or m > 0:
                self.update_results(self.meta is not None)
                if self.step > 0:
                    self.deltas = {}
                    for quant in ['energy', 'forces', 'stress']:
                        self.deltas[quant] = self.results[quant] - pre[quant]
        else:
            self.covlog = f'{float(self.get_covloss().max())}'
        energy = self.results['energy']

        # test
        if self.active and self.test and self.step - self._last_test > self.test:
            self._test()

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

    def veto(self):
        if self.size[0] < 2:
            return False
        if 'forces' in self._veto:
            c1 = abs(self.results['forces']).max() >= self._veto['forces']
        else:
            c1 = False
        if c1:
            self.log('an update is vetoed!')
        return c1

    def update_results(self, retain_graph=False):
        energies = self.cov@self.model.mu
        energy = self.reduce(energies, retain_graph=retain_graph)
        # ... or
        # self.allcov = self.gather(self.cov)
        # energies = self.allcov@self.model.mu
        # retain_graph = self.active or (self.meta is not None)
        # energy = self.reduce(energies, retain_graph=retain_graph, reduced=True)

    def reduce(self, local_energies, op='=', retain_graph=False, reduced=False, is_meta=False):
        energy = local_energies.sum()
        if self.atoms.is_distributed and not reduced:
            torch.distributed.all_reduce(energy)
        forces, stress = self.grads(energy, retain_graph=retain_graph)
        if is_meta:
            forces = nan_to_num(forces, 0.)
        else:
            mean = self.model.mean(self.atoms)
            if mean.grad_fn:  # only constant mean
                raise RuntimeError('mean has grad_fn!')
            energy = energy + mean
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
        #i = self.atoms.first_of_each_atom_type()
        i = self.get_unique_lces()
        inducing = LocalsData([self.atoms.local(j, detach=True) for j in i])
        self.model.set_data(data, inducing)
        # data is stored in _exact, thus we only store the inducing
        if self.tape:
            for loc in inducing:
                self.tape.write(loc)
        details = [(j, self.atoms.numbers[j]) for j in i]
        self.log('seed size: {} {} details: {}'.format(
            *self.size, details))
        if self.tune_for_md:
            self.sample_rand_lces(indices=i, repeat=1)
        self.optimize()

    def get_unique_lces(self, thresh=0.95):
        tmp = (self.atoms.as_ase() if self.to_ase else self.atoms).copy()
        tmp.calc = None
        atoms = TorchAtoms(ase_atoms=tmp)
        atoms.update(posgrad=False, cellgrad=False, dont_save_grads=True,
                     cutoff=self.model.cutoff, descriptors=self.model.descriptors)
        k = self.model.gp.kern(atoms, atoms)
        unique = []
        for i in range(k.shape[0]):
            is_unique = True
            for j in unique:
                if k[i, j] >= thresh:
                    is_unique = False
                    break
            if is_unique:
                unique.append(i)
        return unique

    def sample_rand_lces(self, indices=None, repeat=1, extend_cov=False):
        added = 0
        for _ in range(repeat):
            tmp = (self.atoms.as_ase() if self.to_ase else self.atoms).copy()
            shape = tmp.positions.shape
            tmp.positions += np.random.uniform(-0.05, 0.05, size=shape)
            tmp.calc = None
            atoms = TorchAtoms(ase_atoms=tmp)
            atoms.update(posgrad=False, cellgrad=False, dont_save_grads=True,
                         cutoff=self.model.cutoff, descriptors=self.model.descriptors)
            if indices is None:
                indices = np.random.permutation(len(atoms.loc))
            for k in indices:
                res = abs(self.update_lce(atoms.loc[k]))
                added += res
                if res > 0 and extend_cov:
                    cov = self.model.gp.kern(self.atoms, self.model.X[-1])
                    self.cov = torch.cat([self.cov, cov], dim=1)
        self.log(f'added {added} randomly displaced LCEs')

    def _test(self):
        tmp = self.atoms.as_ase() if self.to_ase else self.atoms
        tmp.set_calculator(self._calc)
        energy = tmp.get_potential_energy()
        forces = tmp.get_forces()
        # write
        self._ktest += 1
        mode = 'a' if self._ktest > 1 else 'w'
        if self.rank == 0:
            ase.io.Trajectory('active_FP.traj', mode).write(tmp)
            tmp.set_calculator(SinglePointCalculator(tmp, **self.results))
            ase.io.Trajectory('active_ML.traj', mode).write(tmp)
        # log
        self.log('testing energy: {}'.format(energy))
        dE = self.results['energy'] - energy
        df = abs(self.results['forces'] - forces)
        self.log('errors (test):  del-E: {:.2g}  max|del-F|: {:.2g}  mean|del-F|: {:.2g}'.format(
            dE, df.max(), df.mean()))
        self._last_test = self.step
        return energy, forces

    def _exact(self, copy):
        tmp = copy.as_ase() if self.to_ase else copy
        tmp.set_calculator(self._calc)
        energy = tmp.get_potential_energy()
        forces = tmp.get_forces()
        if self.tape:
            self.tape.write(tmp)
        self.log('exact energy: {}'.format(energy))
        #
        if self.model.ndata > 0:
            dE = self.results['energy'] - energy
            df = abs(self.results['forces'] - forces)
            self.log('errors (pre):  del-E: {:.2g}  max|del-F|: {:.2g}  mean|del-F|: {:.2g}'.format(
                dE, df.max(), df.mean()))
        self._last_test = self.step
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
        vscale = []
        for z in self.atoms.numbers:
            if z in self.model._vscale:
                vscale.append(self.model._vscale[z])
            else:
                vscale.append(float('inf'))
        vscale = torch.tensor(vscale).sqrt()
        return beta*vscale

    def update_lce(self, loc, beta=None):
        k = None
        if beta is None:
            k = self.model.gp.kern(loc, self.model.X)
            b = self.model.choli@k.detach().t()
            c = (b*b).sum()  # /self.model.gp.kern(loc, loc)
            if loc.number in self.model._vscale:
                vscale = self.model._vscale[loc.number]
            else:
                vscale = float('inf')
            beta = ((1-c)*vscale).clamp(min=0.).sqrt()
            k = k.detach().t()
        added = 0
        m = self.model.indu_counts[loc.number]
        if loc.number in self.model.gp.species:
            if beta >= self.ediff_ub:
                self.model.add_inducing(loc, col=k)
                added = -1 if m < 2 else 1
            elif beta < self.ediff_lb:
                if m < 2 and beta > torch.finfo().eps:
                    self.model.add_inducing(loc, col=k)
                    added = -1
            else:
                ediff = (self.ediff if m > 1
                         else torch.finfo().eps)
                added, delta = self.model.add_1inducing(
                    loc, ediff, detach=False)
        if added != 0:
            if self.model.ridge > 0.:
                self.model.pop_1inducing(clear_cached=True)
                added = 0
            else:
                if self.tape:
                    self.tape.write(loc)
                if self.ioptim == 0:
                    self.optimize()
        return added

    def update_inducing(self):
        added_beta = 0
        added_diff = 0
        added_indices = []
        added_covloss = None
        while True:
            if len(added_indices) == self.atoms.natoms:
                break
            beta = self.get_covloss()
            q = torch.argsort(beta, descending=True)
            for k in q.tolist():
                if k not in added_indices:
                    break
            if beta[k].isclose(torch.ones([])):
                self.blind = True
            loc = self.atoms.local(k, detach=True)
            added = self.update_lce(loc, beta=beta[k])
            if added == 0:
                break
            else:
                if added == -1:
                    self.blind = True
                    added_beta += 1
                elif added == 1:
                    added_diff += 1
                x = self.model.gp.kern(self.atoms, loc)
                self.cov = torch.cat([self.cov, x], dim=1)
                added_indices.append(k)
                added_covloss = beta[k]
        added = added_beta + added_diff
        if added > 0:
            # details = [(k, self.atoms.numbers[k]) for k in added_indices]
            details = ''
            self.log('added indu: {} ({},{}) -> size: {} {} details: {:.2g} {}'.format(
                added, added_beta, added_diff, *self.size, added_covloss, details))
            if not self.normalized:
                self.log(f'kernel diag mean: {self.model.kern_diag_mean}')
            if self.blind:
                self.log('model may be blind -> go robust')
        self.covlog = f'{float(beta[q[0]])}'
        return added

    def update_data(self, try_fake=True):
        # try bypassing
        if self.tune_for_md and len(self.model.data) > 2:
            last = self.model.data[-1]
            if last.natoms == self.atoms.natoms:
                if (last.numbers == self.atoms.numbers).all():
                    if (abs(last.positions-self.atoms.positions) < 0.1).all():
                        return 0

        #
        n = self.model.ndata
        new = self.snapshot(fake=try_fake)
        # self.model.add_1atoms(new, self.ediff_tot, self.fdiff)
        a, de, df = self.model.add_1atoms_fast(new, self.ediff_tot, self.fdiff, self.atoms.xyz,
                                               self.cov, self.atoms.is_distributed)
        added = self.model.ndata - n
        self.log(f'DF: {df}  accept: {added}')
        if added > 0:
            if try_fake:
                self.head()
            self.log('added data: {} -> size: {} {}'.format(
                added, *self.size))
            if self.ioptim in [0, 2]:
                self.optimize()
            elif self.ioptim > 2:
                self._ioptim += 1
                if self._ioptim % (self.ioptim-1) == 0:
                    self.optimize()
                    self._ioptim = 0
        return added

    def optimize(self):
        self.model.optimize_model_parameters(noise_f=self.noise_f)

    def update(self, inducing=True, data=True):
        self.updated = False
        self.get_ready()
        self.blind = False
        m = self.update_inducing() if inducing else 0
        try_real = self.blind or type(self._calc) == SinglePointCalculator
        update_data = (m > 0 and data) or not inducing
        if update_data and not inducing:  # for include_tape
            update_data = self.get_covloss().max() > self.ediff
        n = self.update_data(try_fake=not try_real) if update_data else 0
        if m > 0 or n > 0:
            # TODO: if threshold is reached -> downsizes every time! fix this!
            ch1, ch2 = self.model.downsize(
                self.max_data, self.max_inducing, first=True, lii=True)
            if ch1 or ch2:
                self.log('downsized -> size: {} {}'.format(*self.size))
            if ch2:
                self.cov = self.cov.index_select(1, torch.as_tensor(ch2))
            if self.ioptim == 1:
                self.optimize()
            self.log('fit error (mean,mae): E: {:.2g} {:.2g}   F: {:.2g} {:.2g}   R2: {:.4g}'.format(
                *(float(v) for v in self.model._stats)))
            if self.rank == 0:
                self.log(f'noise: {self.model.scaled_noise}')
                self.log(f'mean: {self.model.mean}')
            if self.pckl:
                self.model.to_folder(self.pckl)
            self.updated = True
        self._update_args = {}
        return m, n

    def include_data(self, data):
        if type(data) == str:
            data = ase.io.read(data, '::')
        tune_for_md = self.tune_for_md
        self.tune_for_md = False
        self.get_ready()

        _calc = self._calc
        for atoms in data:
            if abs(atoms.get_forces()).max() > self.include_params['fmax']:
                continue
            self._calc = atoms.calc
            atoms.set_calculator(self)
            atoms.get_potential_energy()
            atoms.set_calculator(self._calc)
        self._calc = _calc
        self.tune_for_md = tune_for_md

    def include_tape(self, tape, ndata=None):
        if type(tape) == str:
            tape = SgprIO(tape)
        _calc = self._calc
        tune_for_md = self.tune_for_md
        self.tune_for_md = False
        self.get_ready()

        def _save():
            if added_lce[0] > 0:
                if self.ioptim == 1:
                    self.optimize()
                if self.pckl:
                    self.model.to_folder(self.pckl)
                self.log('added lone indus: {}/{} -> size: {} {}'.format(
                    *added_lce, *self.size))
                self.log('fit error (mean,mae): E: {:.2g} {:.2g}   F: {:.2g} {:.2g}   R2: {:.4g}'.format(
                    *(float(v) for v in self.model._stats)))

        #
        added_lce = [0, 0]
        cdata = 0
        for cls, obj in tape.read():
            if cls == 'atoms':
                if abs(obj.get_forces()).max() > self.include_params['fmax']:
                    if len(self.model.data) > 0:
                        continue
                _save()
                self._update_args = dict(inducing=False)
                self._calc = obj.calc
                obj.set_calculator(self)
                obj.get_potential_energy()
                obj.set_calculator(self._calc)
                cdata += 1
                if ndata and cdata >= ndata:
                    break
                added_lce = [0, 0]
            elif cls == 'local':
                obj.stage(self.model.descriptors, True)
                added = self.update_lce(obj)
                added_lce[0] += abs(added)
                added_lce[1] += 1
        _save()

        #
        self._calc = _calc
        self.tune_for_md = tune_for_md

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
        settings = ['ediff', 'ediff_tot', 'fdiff']
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
        kwargs = {'op': '+=', 'reduced': True, 'is_meta': True}
        return energies, kwargs


class ActiveMeta:

    def __init__(self, scale=1e-2):
        self.scale = scale

    def __call__(self, calc):
        cov = calc.gather(calc.cov)
        b = calc.model.choli@cov.t()
        c = (b*b).sum(dim=0)
        beta = (1 - c).clamp(min=0.).sqrt()
        vscale = [calc.model._vscale[z] for z in calc.atoms.numbers]
        vscale = torch.tensor(vscale).sqrt()
        pot = -(beta*vscale).sum()*self.scale
        kwargs = {'op': '+=', 'reduced': True, 'is_meta': True}
        return pot, kwargs

    def update(self):
        pass


def parse_logfile(file='active.log', window=(None, None)):
    start = None
    settings = []
    elapsed = []
    energies = []
    temperatures = []
    covloss = []
    DF = []
    exact_energies = []
    test_energies = []
    indu = []
    errors = []
    test_errors = []
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
            settings = {a: eval(b) for a, b in zip(split[2::2], split[3::2])}

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
        except:
            pass

        if 'meta:' in split:
            meta += [(step, float(split[split.index('meta:')+1]))]

        if 'exact energy' in line:
            exact_energies += [(step, float(split[3]))]

        if 'testing energy' in line:
            test_energies += [(step, float(split[3]))]

        if 'added indu' in line:
            sf = float(split[split.index('details:') + 1])
            indu += [(step, sf)]

        if 'errors (pre)' in line:
            errors += [(step, [float(v) for v in split[4:10:2]])]

        if 'errors (test)' in line:
            test_errors += [(step, [float(v) for v in split[4:10:2]])]

        if 'fit' in line:
            fit += [(step, [float(split[k]) for k in [-7, -6, -4, -3, -1]])]

        if 'DF' in line:
            DF += [(step, float(split[2]), int(split[4]))]

    return energies, exact_energies, test_energies, temperatures, \
        covloss, meta, indu, fit, elapsed, settings, test_errors, DF


def log_to_figure(file, figsize=(10, 5), window=(None, None), meta_ax=True, plot_test=True):
    import pylab as plt
    ml, fp, test, tem, covloss, meta, indu, fit, elapsed,\
        settings, test_errors, DF = parse_logfile(file, window=window)
    fig, _axes = plt.subplots(2, 2, figsize=figsize)
    axes = _axes.reshape(-1)
    # 0
    x, y = zip(*ml)
    axes[0].plot(x, y, label='ML', zorder=1)
    if len(fp) > 0:
        r, s = zip(*fp)
        axes[0].scatter(r, s, color='r', label='ab initio', zorder=2)
    if len(test) > 0:
        r, s = zip(*test)
        axes[0].scatter(r, s, color='g', label='test', zorder=2)
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
    axes[2].set_ylabel('LCE error')
    if len(indu) > 0:
        axes[2].scatter(*zip(*indu), color='lime', label='added', zorder=2)
    axes[2].legend()
    wall = axes[2].twinx()
    wall.plot(*zip(*elapsed), color='cyan', alpha=0.5)
    wall.set_ylabel('minutes')
    axes[2].axhline(y=settings['ediff:'], ls='--', color='k')
    # axes[2].axhline(y=settings['ediff_lb:'], ls='--', color='k')
    # axes[2].axhline(y=settings['ediff_ub:'], ls='--', color='k', alpha=0.3)
    axes[2].grid()
    if len(DF) > 0:
        steps, df, add = zip(*DF)
        axes[2].scatter(steps, df, marker='*', c=add, zorder=2)
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
        axes[3].set_ylabel('MAE')
        # R2
        if q.shape[1] > 4:
            R2 = axes[3].twinx()
            R2.plot(p, 1-q[:, 4], ':', color='grey')
            R2.set_ylabel(r'$1-R^2$')
    if len(test_errors) > 0 and plot_test:
        steps, tmp = zip(*test_errors)
        de, dfmax, df = zip(*tmp)
        axes[3].plot(steps, de, 'v--', color='salmon')
        axes[3].plot(steps, df, 'v--', color='cornflowerblue')
        axes[3].plot(steps, dfmax, 'v:', color='cornflowerblue')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    import sys
    log = sys.argv[1]
    fig = log_to_figure(log)
    fig.savefig(log.replace('.log', '.pdf'))
