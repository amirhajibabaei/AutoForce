
# coding: utf-8

# In[ ]:


from theforce.util.util import iterable
import itertools
from ase.neighborlist import primitive_neighbor_list
import numpy as np
import torch


def mask_values(arr, vals):
    return np.stack([arr == v for v in iterable(vals)]).any(axis=0)


class System:

    def __init__(self, atoms=None, positions=None, cell=None, pbc=None, numbers=None,
                 energy=None, forces=None, max_cutoff=None, cutoff=None):
        """The idea is to store minimal information."""
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

        self.natoms = self.xyz.size(0)
        self.max_cutoff = max_cutoff

        if cutoff is not None:
            self.build_nl(cutoff)

    def build_nl(self, cutoff, self_interaction=False, masks=True):
        i, j, offset = primitive_neighbor_list('ijS', self.pbc, self.cell,
                                               self.xyz.detach().numpy(),
                                               cutoff, numbers=None,
                                               self_interaction=self_interaction)
        cells = torch.from_numpy(np.einsum('ik,kj->ij', offset, self.cell))
        self.r = self.xyz[j] + cells - self.xyz[i]
        self.i = torch.as_tensor(i).long()
        self.j = torch.as_tensor(j).long()
        self.d2 = (self.r**2).sum(dim=-1).view(-1, 1)
        self.d = self.d2.sqrt()
        self.u = self.r / self.d
        self.logd = self.d.log()
        self.logd_deriv = self.u / self.d

        self.mask = {}
        if masks:
            for (a, b) in itertools.product(set(self.nums), set(self.nums)):
                self.get_mask(a, b)
        self.last_cutoff_used = cutoff

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

    def ijr(self, cutoff=None, include=None, ignore_same_numbers=False, self_interaction=False):
        """
        Returns i, j, r (tensors) of neighboring atoms.
        i, j are indices, r is the displacement vector (xyz[j]-xyz[i]).
        If cutoff is None it will use System.max_cutoff.
        If a sequence of atomic numbers is passed as include,
        numbers not in this sequence will be ignored.
        If ignore_same_numbers is True, bonds between similar atoms 
        will be discarded.
        """
        if cutoff is None:
            if self.max_cutoff is None:
                raise RuntimeError('No cutoff found for building neighborlist')
            else:
                cutoff = self.max_cutoff
        if include is None:  # then include all atomic-numbers
            m = np.full(self.natoms, True)
        else:
            m = mask_values(self.nums, include)
        _xyz = self.xyz.detach().numpy()[m]
        i, j, offset = primitive_neighbor_list('ijS', self.pbc, self.cell, _xyz,
                                               cutoff, self_interaction=self_interaction)
        index = np.arange(self.natoms, dtype=np.int)[m]
        i = index[i]
        j = index[j]
        if ignore_same_numbers:
            m2 = self.nums[i] != self.nums[j]
        else:
            m2 = np.full(i.shape[0], True)
        i = torch.from_numpy(i[m2]).long()
        j = torch.from_numpy(j[m2]).long()
        cells = torch.from_numpy(np.einsum('ik,kj->ij', offset[m2], self.cell))
        r = self.xyz[j] + cells - self.xyz[i]
        return i, j, r


class Inducing:

    def __init__(self, X):
        self.X = X

    @property
    def params(self):
        return [s.xyz for s in self.X if s.xyz.requires_grad]

    def __iter__(self):
        for s in self.X:
            yield s

    def __getitem__(self, i):
        return self.X[i]

    def __len__(self):
        return len(self.X)

    def fix_params(self):
        for s in self.X:
            s.xyz.requires_grad = False

    def free_params(self):
        for s in self.X:
            s.xyz.requires_grad = True

    def update_nl_if_requires_grad(self):
        for s in self.X:
            if s.xyz.requires_grad:
                s.build_nl(s.last_cutoff_used)


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
    sys = System(atoms)
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


def test_ijr():
    from ase import Atoms
    xyz = np.random.uniform(0, 10., size=(15, 3))
    nums = np.random.randint(1, 4, size=(15,))
    atoms = Atoms(positions=xyz, numbers=nums,
                  cell=[10, 10, 10], pbc=True)
    sys = System(atoms, cutoff=3.0)
    mask = sys.select(1, 2, bothways=True)
    i, j, r = sys.ijr(cutoff=3.0, include=(1, 2), ignore_same_numbers=True)
    print((sys.nums[i] == sys.nums[sys.i[mask]]).all())
    print((sys.nums[j] == sys.nums[sys.j[mask]]).all())
    d = (r**2).sum(dim=-1).sqrt()
    test_d = d.allclose(sys.d[mask].view(-1))
    print(test_d)
    if not test_d:
        print('probably numbers are just swapped (thats ok!)')
        print(d)
        print(sys.d[mask].view(-1))


if __name__ == '__main__':
    test()
    test_empty_nl()
    test_multi()
    test_ijr()

