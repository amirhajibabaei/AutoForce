#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from theforce.regression.gppotential import PosteriorPotential, PosteriorPotentialFromFolder
from theforce.calculator.posterior import AutoForceCalculator
from theforce.descriptor.atoms import AtomsData, LocalsData, TorchAtoms
from theforce.util.util import date
import torch
import ase


class Leapfrog:

    def __init__(self, dyn, gp, cutoff, calculator=None, ediff=0.1, step=0,
                 train=True, sparse=True, model=None, skip=0, maxfp=None):
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
            self.update_calc(newdata=False)
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

