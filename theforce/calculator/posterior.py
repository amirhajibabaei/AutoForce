
# coding: utf-8

# In[ ]:


import numpy as np
from ase.calculators.calculator import Calculator, all_changes
import torch
from torch.autograd import grad
from theforce.util.util import iterable


class PosteriorCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, potentials, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.potentials = iterable(potentials)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.atoms.xyz.requires_grad = False
        if 'numbers' in system_changes:
            self.atoms.build_loc(self.atoms.loc.rc)
        self.atoms.update()
        energy = torch.stack([pot([self.atoms], 'energy') for pot in
                              self.potentials]).sum(dim=0)
        forces = torch.stack([pot([self.atoms], 'forces') for pot in
                              self.potentials]).sum(dim=0)
        self.results['energy'] = energy.detach().numpy()[0]
        self.results['forces'] = forces.detach().numpy()

