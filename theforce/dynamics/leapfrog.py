# +
from theforce.regression.gppotential import PosteriorPotential, PosteriorPotentialFromFolder
from theforce.calculator.posterior import AutoForceCalculator
from theforce.descriptor.atoms import AtomsData, LocalsData, TorchAtoms
from theforce.util.util import date, class_of_method
import torch
import ase
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
import types
import warnings


def initial_model(gp, atoms, ediff):
    i = atoms.first_of_each_atom_type()
    inducing = LocalsData([atoms.loc[j] for j in i])
    data = AtomsData([atoms])
    model = PosteriorPotential(gp, data, inducing, use_caching=True)
    for j in range(atoms.natoms):
        if j not in i:
            model.add_1inducing(atoms.loc[j], ediff)
    return model


class Leapfrog:

    def __init__(self, dyn, gp, cutoff, ediff=0.1, fdiff=float('inf'), calculator=None, model=None,
                 algorithm='ultrafast', volatile=None, logfile='leapfrog.log', skip=10, skip_volatile=3,
                 undo_volatile=True, correct_verlet=True, tune=(None, None)):
        self.dyn = dyn
        self.gp = PosteriorPotential(gp).gp
        self.cutoff = cutoff
        self._ediff = ediff
        self._fdiff = fdiff
        self.skip = skip
        self.skip_volatile = skip_volatile
        self.undo_volatile = undo_volatile
        self.correct_verlet = correct_verlet
        self._tune = tune

        if type(algorithm) == str:
            self.algorithm = getattr(self, 'algorithm_'+algorithm)
        else:
            self.algorithm = types.MethodType(algorithm, self)

        # atoms
        if type(dyn.atoms) == ase.Atoms:
            self.to_ase = True
            dyn.atoms = TorchAtoms(dyn.atoms)
        else:
            self.to_ase = False
        self.atoms.update(cutoff=cutoff, descriptors=self.gp.kern.kernels)

        # calc
        if calculator:
            self.calculator = calculator
        else:
            self.calculator = dyn.atoms.calc

        # volatile
        self._volatile = volatile if volatile else 2 if model is None else -1

        # initiate
        self.step = 0
        self._fp = []
        self._fp_e = []
        self._ext = []
        self.logfile = logfile
        self.log('leapfrog says Hello!', mode='w')
        self.log('volatile: {}'.format(self._volatile))

        # model
        if model:
            if type(model) == str:
                potential = PosteriorPotentialFromFolder(model)
            else:
                potential = model
            self.log('a model is provided with {} data and {} ref(s)'.format(
                len(potential.data), len(potential.X)))
        else:
            assert ediff is not None
            snap = self.snapshot()
            potential = initial_model(self.gp, snap, ediff)
            potential._cutoff = cutoff
            self.log('update: {}  data: {}  inducing: {}  FP: {}'.format(
                True, len(potential.data), len(potential.inducing), len(self._fp)))
            self.log('a model is initiated with {} data and {} ref(s)'.format(
                len(potential.data), len(potential.X)))
        self.atoms.set_calculator(AutoForceCalculator(potential))
        self.energy = [self.atoms.get_potential_energy()]
        self.temperature = [self.atoms.get_temperature()]

    def detach_writers(self):
        wr = []
        el = []
        for f, i, args, kw in self.dyn.observers:
            if f.__name__ == 'write':
                # for traj
                name = f.__self__.backend.fd.name
                cls = class_of_method(f)  # cls=Trajectory
                wr += [((cls, name), i, args, kw)]
            else:
                el += [(f, i, args, kw)]
        self.dyn.observers = el
        self._writers = wr

    def attach_writers(self):
        for (cls, f), i, args, kw in self._writers:
            self.dyn.observers += [(cls(f, mode='a', atoms=self.dyn.atoms).write,
                                    i, args, kw)]

    @property
    def rank(self):
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        else:
            return 0

    def log(self, mssge, mode='a'):
        if self.rank == 0:
            with open(self.logfile, mode) as f:
                f.write('{} {} {}\n'.format(date(), self.step, mssge))

    @property
    def ediff(self):
        if self._ediff is None:
            return self.model._stats[1] * 2  # dummy number
        else:
            return self._ediff

    @property
    def fdiff(self):
        if self._fdiff is None:
            return self.model._stats[3] * 2  # dummy number
        else:
            return self._fdiff

    @property
    def atoms(self):
        return self.dyn.atoms

    @atoms.setter
    def atoms(self, value):
        self.dyn.atoms = value

    @property
    def model(self):
        return self.atoms.calc.potential

    @property
    def sizes(self):
        return len(self.model.data), len(self.model.X)

    @property
    def fp_nodes(self):
        return self._fp, self._fp_e

    @property
    def ext_nodes(self):
        return self._ext, [self.energy[k] for k in self._ext]

    def volatile(self):
        return len(self._ext) < self._volatile

    def rescale_velocities(self, factor):
        self.atoms.set_velocities(self.atoms.get_velocities()*factor)

    def strain_atoms(self, strain):
        warnings.warn('Leapfrog.strain_atoms is not robust!')
        cell = (np.eye(3) + strain) @ self.atoms.cell.T
        self.atoms.set_cell(cell, scale_atoms=True)

    def rescale_cell(self, f):
        self.atoms.set_cell(f*self.atoms.cell, scale_atoms=True)

    def snapshot(self, fake=False, copy=None):
        if copy is None:
            copy = self.atoms.copy()
        if fake:
            energy = self.atoms.get_potential_energy()
            forces = self.atoms.get_forces()
        else:
            if self.to_ase:
                tmp = copy.as_ase()
            else:
                tmp = copy
            tmp.set_calculator(self.calculator)
            energy = tmp.get_potential_energy()
            forces = tmp.get_forces()
            ase.io.Trajectory('_FP.traj', 'a').write(tmp)
            self._fp.append(self.step)
            self._fp_e.append(energy)
            self.log('exact energy: {}'.format(energy))
        copy.set_calculator(SinglePointCalculator(copy, energy=energy,
                                                  forces=forces))
        copy.set_targets()
        return copy

    def algorithm_robust(self, datafirst=True):
        new = self.snapshot()
        if datafirst is None:
            datafirst = np.random.choice([True, False])
        if datafirst:
            self.model.add_1atoms(new, self.ediff, self.fdiff)
        locs = new.loc
        added_refs, _ = self.model.add_ninducing(
            locs, self.ediff, descending=False)
        if not datafirst:
            self.model.add_1atoms(new, self.ediff, self.fdiff)

    def algorithm_fast(self):
        locs = self.atoms.calc.atoms.loc
        added_refs, _ = self.model.add_ninducing(
            locs, self.ediff, descending=False)
        if added_refs > 0:
            new = self.snapshot()
            self.model.add_1atoms(new, self.ediff, self.fdiff)

    def algorithm_fastfast(self):
        locs = self.atoms.calc.atoms.loc
        added_refs, change = self.model.add_ninducing(locs, self.ediff)
        self.log('added refs: {}  ediff at break: {}'.format(
            added_refs, change))
        if added_refs > 0:
            new = self.snapshot()
            self.model.add_1atoms(new, self.ediff, self.fdiff)

    def algorithm_ultrafast(self):
        locs = self.atoms.calc.atoms.loc
        added_refs, change = self.model.add_ninducing(locs, self.ediff)
        self.log('added refs: {}  ediff at break: {}'.format(
            added_refs, change))
        if added_refs > 0:
            a = len(self.model.data)
            new = self.snapshot(fake=True)
            de, df = self.model.add_1atoms(new, self.ediff, self.fdiff)
            if len(self.model.data) > a:
                self.model.pop_1data(clear_cached=True)
                new = self.snapshot(copy=new)
                self.model.add_1atoms(new, self.ediff, self.fdiff)

    def update_model(self):
        forces_before = self.atoms.get_forces()
        # undo if previous update is not necessary
        if self.volatile() and self._fp[-1] > 0 and self.undo_volatile:
            if self._fp[-1] != (self._ext[-1] if len(self._ext) > 0 else 0):
                if self.step_plus != self._fp[-1]:
                    warnings.warn('step_plus != fp[-1]')
                else:
                    if self.data_plus > 0 or self.ref_plus > 0:
                        self.undo_update()
                        self.log('undo: {}  data: {}  inducing: {}'.format(
                            self._fp[-1], *self.sizes))
        # update
        size1 = self.sizes
        if self.volatile():
            self.algorithm_robust()
        else:
            self.algorithm()
        size2 = self.sizes
        self.data_plus = size2[0]-size1[0]
        self.ref_plus = size2[1]-size1[1]
        self.step_plus = self.step
        tf = self.data_plus > 0 or self.ref_plus > 0
        if tf:
            self.atoms.calc.results.clear()
            if self.correct_verlet:
                self.verlet(forces_before)
            if not self.volatile() and not self.model.is_well():
                self.model.tune_noise(*self._tune)
                e, ev, f, fv = (float(v) for v in self.model._stats)
                self.log(
                    f'tuning noise: {self.model.gp.noise.signal}  stats: {(e, ev)}  {(f, fv)}')
        return tf

    def undo_update(self):
        d = self.data_plus
        i = self.ref_plus
        while d > 0:
            self.model.pop_1data()
            d -= 1
        while i > 0:
            self.model.pop_1inducing()
            i -= 1

    def verlet(self, forces_before):
        atoms = self.atoms
        f = forces_before
        p = atoms.get_momenta()
        p += 0.5 * self.dyn.dt * f
        masses = atoms.get_masses()[:, np.newaxis]
        r = atoms.get_positions()
        atoms.set_positions(r + self.dyn.dt * p / masses)
        if atoms.constraints:
            p = (atoms.get_positions() - r) * masses / self.dt
        atoms.set_momenta(p, apply_constraint=False)
        f = atoms.get_forces(md=True)
        atoms.set_momenta(atoms.get_momenta() + 0.5 * self.dyn.dt * f)
        return f

    def doit(self, prob=1):

        # check
        ext = False
        if len(self.energy) >= 3:
            d1 = self.energy[-1] - self.energy[-2]
            d2 = self.energy[-2] - self.energy[-3]
            if d1*d2 < 0:
                ext = True
                # unless it's a artificial ext!
                if len(self._fp) > 0 and self.step - self._fp[-1] < 3:
                    ext = False
        # decide
        last = 0 if len(self._fp) == 0 else self._fp[-1]
        if ext:
            self.log('extremum')
            self._ext += [self.step]
            if not self.volatile() and self._ext[-1]-last < self.skip:
                return False
            return np.random.choice([True, False], p=[prob, 1-prob])  # main
        else:
            if self.volatile() and ((self.step == 0 and len(self._fp) == 0)
                                    or self.step-last > self.skip_volatile):
                return True
            return False  # main

    def run(self, maxsteps, prob=1):
        for _ in range(maxsteps):
            if prob > 0 and self.doit(prob=prob):
                self.log('updating ...')
                self.log('update: {}  data: {}  inducing: {}  FP: {}'.format(
                    self.update_model(), *self.sizes, len(self._fp)))
            self.dyn.run(1)
            self.step += 1
            self.energy += [self.atoms.get_potential_energy()]
            self.temperature += [self.atoms.get_temperature()]
            self.log('{} {}'.format(self.energy[-1], self.temperature[-1]))

    def run_updates(self, maxupdates, prob=1):
        updates = 0
        steps = 0
        stresses = []
        while updates < maxupdates:
            if prob > 0 and self.doit(prob=prob):
                self.log('updating ...')
                self.log('update: {}  data: {}  inducing: {}  FP: {}'.format(
                    self.update_model(), *self.sizes, len(self._fp)))
                updates += 1
            self.dyn.run(1)
            self.step += 1
            steps += 1
            self.energy += [self.atoms.get_potential_energy()]
            stresses += [self.atoms.get_stress().reshape(1, -1)]
            self.temperature += [self.atoms.get_temperature()]
            self.log('{} {}'.format(self.energy[-1], self.temperature[-1]))
        steps_per_update = steps / updates
        average_energy = np.array(self.energy[-steps:]).mean()
        average_temp = np.array(self.temperature[-steps:]).mean()
        self.log('steps per update: {}, energy: {}, temperature: {}'.format(
            steps_per_update, average_energy, average_temp))
        stress = np.concatenate(stresses).mean(axis=0)
        self.log('stress: {}'.format(stress))
        return steps_per_update, average_energy, average_temp, stress
