
# coding: utf-8

# In[ ]:


import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
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
        self.params = [par for term in self.terms for par in term.parameters()]
        self.with_grads = False

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
            for pot in self.terms:
                energy = energy + pot(nums[a], nums[neighbors], r)
        forces, = grad(-energy, xyz, create_graph=self.with_grads)

        if self.with_grads:
            xyz.detach_()
        else:
            energy = energy.detach().numpy().reshape(1)[0]
            forces = forces.detach().numpy()

        self.results['energy'] = energy
        self.results['forces'] = forces


def _loss_function(eng, systems):
    # data
    energies = torch.tensor([atoms.get_potential_energy()
                             for atoms in systems])
    forces = torch.cat([torch.tensor(atoms.get_forces())
                        for atoms in systems]).view(-1)

    # pred
    _energies = []
    _forces = []
    for atoms in systems:
        eng.calculate(atoms)
        _energies += [eng.results['energy']]
        _forces += [eng.results['forces']]
    _energies = torch.tensor(_energies).view(-1)
    _forces = torch.cat(_forces).view(-1)

    # loss
    loss = ((energies - _energies)**2).sum() + ((forces - _forces)**2).sum()
    return loss


def _train_engine(eng, systems, steps=10, lr=0.01):
    """
    _loss_function and _train_engine are only created for testing
    and most likely they will be useless in real applications.
    gradient descent algorithms are very inefficient in this way.
    """
    if not hasattr(eng, 'optimizer'):
        eng.optimizer = torch.optim.Adam(eng.params, lr=lr)

    for _ in range(steps):
        def closure():
            eng.optimizer.zero_grad()
            loss = _loss_function(eng, systems)
            loss.backward()
            return loss
        eng.optimizer.step(closure)
    print('trained for {} steps, and loss is {}'.format(steps, closure()))
    eng.optimizer.zero_grad()


def example():
    #from theforce.calculator.engine import Engine
    from torch.nn import Module, Parameter

    class LJ(Module):

        def __init__(self,):
            super().__init__()
            self.eps = Parameter(torch.tensor(1.0), requires_grad=False)
            self.sigma = Parameter(torch.tensor(1.0), requires_grad=False)

        def forward(self, a, aa, r):
            x = (self.sigma**2/(r**2).sum(dim=-1))
            return 4*self.eps*(x**6 - x**3).sum()/2

        def extra_repr(self):
            print(self.eps)
            print(self.sigma)

    from ase import Atoms
    from ase.optimize import BFGS

    atoms = Atoms('ArAr', positions=[[0, 0, 0], [2., 0, 0]], cell=[
                  9., 9., 9.], pbc=True)
    calc = Engine(rc=3.0, terms=LJ())
    atoms.set_calculator(calc)
    BFGS(atoms).run(fmax=1e-5)
    dmin = ((atoms.positions[0]-atoms.positions[1])**2).sum()
    print('isclose: {} (diff = {})'.format(
        np.isclose(dmin, 2**(1./3)), dmin-2**(1./3)))


if __name__ == '__main__':
    example()

