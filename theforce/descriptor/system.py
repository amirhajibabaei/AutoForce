#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from ase.neighborlist import primitive_neighbor_list
import itertools
# torch.set_default_tensor_type(torch.DoubleTensor)


def dict_to_indices(numbers, elements=None):
    if elements is None:
        elements = sorted(list(set(numbers.tolist())))
    elements_dict = {e: i for i, e in enumerate(elements)}
    return torch.tensor(list(map(lambda a: elements_dict[a], numbers.tolist())))


class System:

    def __init__(self, atoms=None, positions=None, cell=None, pbc=None, numbers=None,
                 energy=None, forces=None, elements=None, cutoff=None):

        if atoms:
            self.xyz = torch.as_tensor(atoms.positions)
            self.cell = atoms.cell
            self.pbc = atoms.pbc
            self.nums = atoms.get_atomic_numbers()
            try:
                self.forces = torch.as_tensor(atoms.get_forces())
            except RuntimeError:
                self.forces = None
            try:
                self.energy = torch.tensor(atoms.get_potential_energy())
            except RuntimeError:
                self.energy = None
        else:
            self.xyz = torch.as_tensor(positions)
            self.cell = cell
            self.pbc = pbc
            self.nums = (numbers if numbers is not None else
                         np.zeros(self.xyz.size(0), dtype=np.int))
            self.forces = (torch.as_tensor(forces)
                           if forces is not None else None)
            self.energy = (torch.as_tensor(energy)
                           if energy is not None else None)

        self.idx = dict_to_indices(self.nums, elements=elements)
        self.natoms = self.xyz.size(0)

        if cutoff is not None:
            self.build_nl(cutoff)

    def build_nl(self, cutoff, self_interaction=False, masks=True):
        i, j, offset = primitive_neighbor_list('ijS', self.pbc, self.cell,
                                               self.xyz.detach().numpy(),
                                               cutoff, numbers=None,
                                               self_interaction=self_interaction)
        if i.shape[0] > 0:
            self.r = self.xyz[j] + torch.einsum('ik,kj->ij',
                                                torch.as_tensor(
                                                    offset.astype(np.float)),
                                                torch.as_tensor(self.cell)) - self.xyz[i]
        else:
            self.r = torch.zeros(0, 3)

        self.i = torch.as_tensor(i).long()
        self.j = torch.as_tensor(j).long()
        self.d = (self.r**2).sum(dim=-1).sqrt().view(-1, 1)
        self.dr = self.r / self.d

        self.mask = {}
        if masks:
            for (a, b) in itertools.product(set(self.nums), set(self.nums)):
                self.get_mask(a, b)

    def get_mask(self, a, b):
        try:
            return self.mask[(a, b)]
        except KeyError:
            mask = torch.tensor(np.logical_and(self.nums[self.i] == a,
                                               self.nums[self.j] == b))
            self.mask[(a, b)] = mask
            return mask

    def select(self, a, b, bothways=True):
        m = self.get_mask(a, b)
        if bothways and a != b:
            m = (m.byte() | self.get_mask(b, a).byte()).to(torch.bool)
        return m


def test():
    from ase.io import Trajectory
    import numpy as np
    from theforce.util.flake import hexagonal_flake
    from ase.calculators.lj import LennardJones
    from ase import Atoms
    cell = np.array([9., 9., 9.])
    positions = hexagonal_flake(a=1.1, centre=True) + cell/2
    atoms = Atoms(positions=positions, cell=cell, pbc=True)
    atoms.set_calculator(LennardJones(epsilon=1.0, sigma=1.0, rc=3.0))
    atoms.get_potential_energy()
    sys = System(atoms, elements=[0])
    sys.xyz.requires_grad = True
    sys.build_nl(3.0)
    r2 = (sys.r**2).sum(dim=-1)
    energy = 2*((1.0/r2)**6-(1.0/r2)**3).sum()
    energy.backward()
    print(torch.allclose(sys.forces, -sys.xyz.grad))


def test_empty_nl():
    from ase import Atoms
    atoms = Atoms(positions=[[0, 0, 0], [3.1, 0, 0]],
                  cell=[10., 10., 10.], pbc=True)
    sys = System(atoms, cutoff=3.0)


def test_multi():
    from ase import Atoms
    xyz = np.random.uniform(0, 10., size=(15, 3))
    nums = np.random.randint(1, 4, size=(15,))
    atoms = Atoms(positions=xyz, numbers=nums,
                  cell=[10, 10, 10], pbc=True)
    sys = System(atoms, cutoff=3.0)
    mask = sys.select(2, 1, bothways=False)
    assert (sys.nums[sys.i[mask]] == 2).all() and (
        sys.nums[sys.j[mask]] == 1).all()

    mask = sys.select(2, 1, bothways=True)
    # print(sys.nums[sys.i])
    # print(sys.nums[sys.j])
    # print(mask.byte().numpy())


if __name__ == '__main__':
    test()
    test_empty_nl()
    test_multi()

