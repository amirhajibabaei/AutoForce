
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from ase.neighborlist import primitive_neighbor_list
# torch.set_default_tensor_type(torch.DoubleTensor)


def dict_to_indices(numbers, elements=None):
    if elements is None:
        elements = sorted(list(set(numbers.tolist())))
    elements_dict = {e: i for i, e in enumerate(elements)}
    return torch.tensor(list(map(lambda a: elements_dict[a], numbers.tolist())))


class System:

    def __init__(self, atoms=None, positions=None, cell=None, pbc=None, numbers=None,
                 energy=None, forces=None, elements=None):

        if atoms:
            self.xyz = torch.from_numpy(atoms.positions)
            self.cell = atoms.cell
            self.pbc = atoms.pbc
            self.nums = atoms.get_atomic_numbers()
            try:
                self.forces = torch.from_numpy(atoms.get_forces())
            except RuntimeError:
                self.forces = None
            try:
                self.energy = torch.tensor(atoms.get_potential_energy())
            except RuntimeError:
                self.energy = None
        else:
            self.xyz = torch.from_numpy(positions)
            self.cell = cell
            self.pbc = pbc
            self.nums = (numbers if numbers else
                         np.zeros(self.xyz.size(0), dtype=np.int))
            self.forces = (torch.from_numpy(forces) if forces else None)
            self.energy = (torch.tensor(energy) if energy else None)

        self.idx = dict_to_indices(self.nums, elements=elements)
        self.natom = self.xyz.size(0)

    def build_nl(self, cutoff, self_interaction=False):
        self.i, self.j, offset = primitive_neighbor_list('ijS', self.pbc, self.cell,
                                                         self.xyz.detach().numpy(),
                                                         cutoff, numbers=None,
                                                         self_interaction=self_interaction)
        self.r = self.xyz[self.j] + torch.einsum('ik,kj->ij',
                                                 torch.from_numpy(
                                                     offset.astype(np.float)),
                                                 torch.from_numpy(self.cell)) - self.xyz[self.i]


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


if __name__ == '__main__':
    test()

