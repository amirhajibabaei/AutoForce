
# coding: utf-8

# In[ ]:


import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
#from ase.calculators.calculator import PropertyNotImplementedError
import torch
from torch.autograd import grad
from theforce.util.util import iterable


class Engine(Calculator):
    implemented_properties = ['energy', 'forces']
    default_parameters = {'rc': None}

    def __init__(self, terms=None, ** kwargs):
        Calculator.__init__(self, **kwargs)
        if self.parameters.rc is None:
            raise NotImplementedError('pass a cutoff!')
        self.terms = iterable(terms)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)
        if 'numbers' in system_changes:
            rc = self.parameters.rc
            self.nl = NeighborList([rc / 2] * natoms, self_interaction=False,
                                   bothways=True)
        self.nl.update(self.atoms)

        xyz = torch.tensor(self.atoms.positions, requires_grad=True)
        nums = self.atoms.numbers
        cell = self.atoms.cell
        energy = 0.0
        for a in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(a)
            cells = torch.from_numpy(np.dot(offsets, cell))
            r = xyz[neighbors] + cells - xyz[a]
            energy = energy + sum([pot(nums[a], nums[neighbors], r)
                                   for pot in self.terms])
        forces, = grad(energy, xyz)

        self.results['energy'] = energy.detach().numpy()
        self.results['forces'] = forces.detach().numpy()

