
# coding: utf-8

# In[ ]:


import numpy as np
from ase.calculators.calculator import Calculator, all_changes
import torch
from torch.autograd import grad
from theforce.util.util import iterable
import warnings


class PosteriorCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy']

    def __init__(self, potentials, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.potentials = iterable(potentials)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.atoms.update()
        energy = torch.stack([pot([self.atoms], 'energy') for pot in
                              self.potentials]).sum(dim=0)
        forces = torch.stack([pot([self.atoms], 'forces') for pot in
                              self.potentials]).sum(dim=0)
        self.results['energy'] = energy.detach().numpy()[0]
        self.results['forces'] = forces.detach().numpy()
        # NOTE: check if this is correct!
        self.results['free_energy'] = self.results['energy']


class PosteriorStressCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy', 'stress']

    def __init__(self, potentials, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.potentials = iterable(potentials)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.atoms.update(cellgrad=True, forced=True)
        energy = torch.stack([pot([self.atoms], 'energy', enable_grad=True) for pot in
                              self.potentials]).sum(dim=0)
        forces = torch.stack([pot([self.atoms], 'forces') for pot in
                              self.potentials]).sum(dim=0)

        # stress
        stress1 = (forces[:, None]*self.atoms.xyz[..., None]).sum(dim=0)
        cellgrad, = grad(energy, self.atoms.lll)
        stress2 = (cellgrad[:, None]*self.atoms.lll[..., None]).sum(dim=0)
        stress = (stress1 + stress2).detach().numpy() / self.atoms.get_volume()

        self.results['energy'] = energy.detach().numpy()[0]
        self.results['forces'] = forces.detach().numpy()
        self.results['free_energy'] = self.results['energy']
        self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]


class PosteriorVarianceCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy']

    def __init__(self, potential, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.potential = potential

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.atoms.update()
        energy, energy_var = self.potential(
            [self.atoms], 'energy', variance=True)
        forces, forces_var = self.potential(
            [self.atoms], 'forces', variance=True)
        self.results['energy'] = energy.detach().numpy()[0]
        self.results['forces'] = forces.detach().numpy()
        # variances
        self.results['energy_var'] = energy_var.detach().numpy()[0]
        self.results['forces_var'] = forces_var.detach().numpy()
        # NOTE: check if this is correct!
        self.results['free_energy'] = self.results['energy']

