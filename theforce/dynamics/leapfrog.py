#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from theforce.regression.gppotential import PosteriorPotential, PosteriorPotentialFromFolder
from theforce.calculator.posterior import AutoForceCalculator
from theforce.descriptor.atoms import AtomsData, LocalsData, TorchAtoms
from theforce.util.util import date
import torch
import ase
import numpy as np


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

    def __init__(self, dyn, gp, cutoff, ediff=0.1, fdiff=float('inf'), calculator=None, model=None, init=None):
        self.dyn = dyn
        self.gp = gp
        self.cutoff = cutoff
        self.ediff = ediff
        self.fdiff = fdiff

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

        # init
        self.init = True if model is None else False if init is None else init

        # model
        self.step = 0
        self._fp = []
        self._fp_e = []
        self._ext = []
        self.log('leapfrog says Hello!'.format(date()), mode='w')
        if model:
            if type(model) == str:
                potential = PosteriorPotentialFromFolder(model)
            else:
                potential = model
        else:
            snap = self.snapshot()
            potential = initial_model(self.gp, snap, self.ediff)
        self.atoms.set_calculator(AutoForceCalculator(potential))
        self.energy = [self.atoms.get_potential_energy()]
        self.temperature = [self.atoms.get_temperature()]

    def log(self, mssge, file='leapfrog.log', mode='a'):
        with open(file, mode) as f:
            f.write('{} {} {}\n'.format(date(), self.step, mssge))

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

    def snapshot(self):
        tmp = self.atoms.copy()
        if self.to_ase:
            tmp = tmp.as_ase()
        tmp.set_calculator(self.calculator)
        self._fp.append(self.step)
        self._fp_e.append(tmp.get_potential_energy())
        tmp.get_forces()
        ase.io.Trajectory('_FP.traj', 'a').write(tmp)
        if self.to_ase:
            tmp = TorchAtoms(ase_atoms=tmp, cutoff=self.cutoff,
                             descriptors=self.gp.kern.kernels)
        else:
            tmp.set_targets()
        tmp.single_point()
        return tmp

    def update_model(self):
        self.size1 = self.sizes
        new = self.snapshot()
        self.model.add_1atoms(new, self.ediff, self.fdiff)
        for loc in new.loc:
            ediff = self.ediff if self.sizes[1] > 1 else torch.finfo().tiny
            self.model.add_1inducing(loc, ediff)
        self.size2 = self.sizes

    def undo_update(self):
        d = self.size2[0]-self.size1[0]
        i = self.size2[1]-self.size1[1]
        while d > 0:
            self.model.pop_1data()
            d -= 1
        while i > 0:
            self.model.pop_1inducing()
            i -= 1

    def doit(self, prob=1):

        # check
        ext = False
        if len(self.energy) >= 3:
            d1 = self.energy[-1] - self.energy[-2]
            d2 = self.energy[-2] - self.energy[-3]
            if d1*d2 < 0:
                ext = True

        # decide
        if ext:
            self._ext += [self.step]
            if len(self._ext) > 2 and self._ext[-1]-self._fp[-1] < 10:
                return False
            return np.random.choice([True, False], p=[prob, 1-prob])  # main
        else:
            last = 0 if len(self._fp) == 0 else self._fp[-1]
            if self.init and len(self._ext) <= 2 and self.step-last > 3:
                return True
            return False  # main

    def run(self, maxsteps, prob=1):
        for _ in range(maxsteps):
            if prob > 0 and self.doit(prob=prob):
                self.log('updating ...')
                self.update_model()
                self.log('data: {} inducing: {}'.format(*self.sizes))
            self.dyn.run(1)
            self.step += 1
            self.energy += [self.atoms.get_potential_energy()]
            self.temperature += [self.atoms.get_temperature()]
            self.log('{} {}'.format(self.energy[-1], self.temperature[-1]))

    def run_updates(self, maxupdates, prob=1):
        updates = 0
        steps = 0
        while updates < maxupdates:
            if prob > 0 and self.doit(prob=prob):
                self.log('updating ...')
                self.update_model()
                self.log('data: {} inducing: {}'.format(*self.sizes))
                updates += 1
            self.dyn.run(1)
            self.step += 1
            steps += 1
            self.energy += [self.atoms.get_potential_energy()]
            self.temperature += [self.atoms.get_temperature()]
            self.log('{} {}'.format(self.energy[-1], self.temperature[-1]))
        steps_per_update = steps / updates
        average_temp = np.array(self.temperature[-steps:]).mean()
        self.log('steps per update: {}, temperature: {}'.format(
            steps_per_update, average_temp))
        return steps_per_update, average_temp


class _Leapfrog:

    def __init__(self, dyn, gp, cutoff, calculator=None, ediff=0.1, step=0,
                 train=True, sparse=True, model=None, skip=0, maxfp=None, firstfp=False):
        self.dyn = dyn
        self.gp = gp
        self.cutoff = cutoff
        self.train = train
        self.sparse = sparse
        self.ediff = ediff

        self.skip = skip
        self.nfp = 0
        self.maxfp = maxfp

        if type(dyn.atoms) == ase.Atoms:
            self.to_ase = True
            dyn.atoms = TorchAtoms(dyn.atoms)
        else:
            self.to_ase = False
        if calculator:
            self.exact = calculator
        else:
            self.exact = self.atoms.calc

        self.step0 = step
        self.step = self.step0
        self.log('leapfrog says Hello!'.format(date()), mode='w')
        self.atoms.update(cutoff=cutoff, descriptors=self.gp.kern.kernels)
        if model:
            if type(model) == str:
                self.model = PosteriorPotentialFromFolder(model)
            else:
                self.model = model
            self.data = self.model.data
            self.inducing = self.model.X
            self.update_calc(newdata=firstfp)
            self.trigger = self.skip
        else:
            self.model = None
            self.data = AtomsData(X=[])
            self.inducing = LocalsData(X=[])
            self.update_calc()
        self.energy = [self.atoms.get_potential_energy()]

    @property
    def atoms(self):
        return self.dyn.atoms

    @atoms.setter
    def atoms(self, value):
        self.dyn.atoms = value

    @property
    def steps(self):
        return torch.arange(self.step0, self.step+1).numpy()

    @property
    def nodes(self):
        return self.node, [self.energy[k-self.step0] for k in self.node]

    def log(self, mssge, file='leapfrog.log', mode='a'):
        with open(file, mode) as f:
            f.write('{} {} {}\n'.format(date(), self.step, mssge))

    def update_calc(self, newdata=True):
        self.log("updating ...")
        if newdata:
            new = self.get_exact_data()
            new.update(cutoff=self.cutoff, descriptors=self.gp.kern.kernels)
            self.add_data(new)
            self.add_locals(new)
        self.create_node()
        self.atoms.set_calculator(
            AutoForceCalculator(self.model))
        self.log("new calculator")

    def get_exact_data(self):
        if not hasattr(self, 'exact_results'):
            self.exact_results = ase.io.Trajectory('exact_calcs.traj', 'w')
        tmp = self.atoms.copy()
        if self.to_ase:
            tmp = tmp.as_ase()
        tmp.set_calculator(self.exact)
        tmp.get_forces()
        self.exact_results.write(tmp)
        if self.to_ase:
            tmp = TorchAtoms(ase_atoms=tmp)
        else:
            tmp.set_targets()
        tmp.single_point()
        self.log("exact calculation")
        self.nfp += 1
        return tmp

    def add_data(self, new):
        self.data += AtomsData(X=[new])
        if self.model is not None:
            self.update_model()
        self.log("data size: {}".format(len(self.data)))

    def update_model(self, log=True):
        if self.model:
            self.model.set_data(self.data, self.inducing, use_caching=True)
        else:
            self.model = PosteriorPotential(self.gp, self.data, inducing=self.inducing,
                                            use_caching=True)
        if log:
            self.log("new model")

    def add_locals(self, atms, measure='energy'):
        ind = AtomsData([atms]).to_locals()
        ind.stage(descriptors=self.gp.kern.kernels)
        if self.sparse:
            if self.model is None:
                self.inducing += ind[0]
                self.update_model(log=False)
            # sort
            queue = getattr(self, 'sort_{}_measure'.format(measure))(ind)
            # add
            e0 = self.model([atms])
            for i in queue:
                self.inducing += ind[i]
                self.update_model(log=False)
                e = self.model([atms])
                if abs(e - e0) < self.ediff:
                    del self.inducing.X[-1]
                    break
                e0 = e
        else:
            self.inducing += ind
        self.update_model()
        self.log("inducing size: {}".format(len(self.inducing)))

    def sort_variance_measure(self, ind):
        e, v = self.model(ind, variance=True)
        descending = torch.argsort(v, descending=True).tolist()
        return descending

    def sort_energy_measure(self, ind):
        # sort
        e = self.model(ind)
        delta = []
        for e1, loc in zip(*[e, ind]):
            self.inducing += loc
            self.update_model(log=False)
            e2 = self.model(loc)
            delta += [abs(e2 - e1)]
            del self.inducing.X[-1]
        descending = torch.argsort(
            torch.cat(delta), descending=True).tolist()
        self.update_model(log=False)
        return descending

    def create_node(self):
        try:
            self.node += [self.step]
            self.node_atoms += [self.atoms.copy()]
        except AttributeError:
            self.node = [self.step]
            self.node_atoms = [self.atoms.copy()]

    def _doit(self):
        if not self.train:
            return False

        if self.step-self.node[-1] < 3:
            return False

        try:
            d1 = self.energy[-1] - self.energy[-2]
            d2 = self.energy[-2] - self.energy[-3]
            if d1*d2 < 0:  # "<" instead of "<=" for E=Constant case
                if not hasattr(self, "trigger"):
                    self.trigger = 0
                    return True
                else:
                    self.trigger += 1
                    if self.trigger > self.skip:
                        self.trigger = 0
                        return True
                    else:
                        return False
        except:
            pass

        # alert: dummy number 10
        if not hasattr(self, "trigger") and self.step-self.node[-1] > 10:
            return True

        return False

    def doit(self):
        if self.maxfp is None:
            return self._doit()
        else:
            if self.nfp < self.maxfp:
                return self._doit()
            else:
                return False

    def run(self, maxsteps):
        for _ in range(maxsteps):
            if self.doit():
                self.update_calc()
            self.dyn.run(1)
            self.step += 1
            self.energy += [self.atoms.get_potential_energy()]

