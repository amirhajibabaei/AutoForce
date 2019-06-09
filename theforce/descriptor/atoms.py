
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from torch import ones_like, as_tensor, from_numpy, cat
from ase.atoms import Atoms
from ase.neighborlist import NeighborList
from ase.calculators.calculator import PropertyNotImplementedError


class Local:

    def __init__(self, i, j, a, b, r, descriptors=[]):
        """
        i, j: indices
        a, b: atomic numbers
        r : r[j] - r[i]
        """
        self._i = from_numpy(np.full_like(j, i))
        self._j = from_numpy(j)
        self._a = from_numpy(np.full_like(b, a))
        self._b = from_numpy(b)
        self._r = r
        self._m = ones_like(self._i).to(torch.bool)
        for desc in descriptors:
            value, grad, indices = desc.calculate(self)
            setattr(self, desc.name+'_value', value)
            setattr(self, desc.name+'_grad', grad)
            setattr(self, desc.name+'_indices', indices)

    @property
    def i(self):
        return self._i[self._m]

    @property
    def j(self):
        return self._j[self._m]

    @property
    def a(self):
        return self._a[self._m]

    @property
    def b(self):
        return self._b[self._m]

    @property
    def r(self):
        return self._r[self._m]

    def select(self, a, b, bothways=False, in_place=True):
        m = (self._a == a) & (self._b == b)
        if bothways and a != b:
            m = (m | ((self._a == b) & (self._b == a)))
        if not bothways and a == b:
            m = (m & (self._j > self._i))
        if in_place:
            self._m[:] = m.to(torch.bool)
        return m.to(torch.bool)

    def unselect(self):
        self._m[:] = True


class LocalEnvirons(NeighborList, list):

    def __init__(self, atoms, setup):
        """
        This can be used instead of Neighborlist in calculators.
        """
        self.atoms = atoms
        cutoffs = atoms.natoms*[setup.rc / 2]
        super().__init__(cutoffs, skin=0.0, self_interaction=False, bothways=True)
        self.descriptors = setup.descriptors
        self._copy = np.zeros_like(atoms.positions)

    def update(self):
        if not np.allclose(self.atoms.positions, self._copy):
            self._copy[:] = self.atoms.positions[:]
            super().update(self.atoms)
            self.loc = []
            types = self.atoms.get_atomic_numbers()
            for a in range(self.atoms.natoms):
                n, off = self.nl.get_neighbors(a)
                cells = from_numpy(np.dot(off, self.atoms.cell))
                r = self.atoms.xyz[n] - self.atoms.xyz[a] + cells
                self.loc += [Local(a, n, types[a], types[n],
                                   r, self.descriptors)]

    def select(self, a, b, bothways=False, in_place=True):
        return torch.cat([env.select(a, b, bothways=bothways, in_place=in_place)
                          for env in self.loc])

    def unselect(self):
        for env in self.loc:
            env.unselect()

    def __getitem__(self, k):
        return self.loc[k]

    def __iter__(self):
        for env in self.loc:
            yield env

    def __getattr__(self, attr):
        return torch.cat([getattr(env, attr) for env in self.__dict__['loc']])


class TorchAtoms(Atoms):

    def __init__(self, setup, ase_atoms=None, energy=None, forces=None, **kwargs):
        super().__init__(**kwargs)

        if ase_atoms:
            self.arrays = ase_atoms.arrays
        self.xyz = from_numpy(self.positions)

        try:
            self.target_energy = as_tensor(energy)
            self.target_forces = as_tensor(forces)
        except RuntimeError:
            try:
                self.target_energy = as_tensor(
                    ase_atoms.get_potential_energy())
                self.target_forces = as_tensor(ase_atoms.get_forces())
            except AttributeError or PropertyNotImplementedError:
                pass

        self.loc = LocalEnvirons(self, setup)
        self.loc.update()

    @property
    def natoms(self):
        return self.get_number_of_atoms()

    def __getattr__(self, attr):
        return getattr(self.__dict__['loc'], attr)


class Setup:

    def __init__(self, rc, descriptors):
        self.rc = rc
        self.descriptors = descriptors
        for i, desc in enumerate(self.descriptors):
            desc.name = 'desc_{}'.format(i)


class DummyPairDistance:
    def __init__(self, a, b, kernel=None):
        self.a = a
        self.b = b
        self.kernel = kernel

    def calculate(self, loc):
        loc.select(self.a, self.b)
        d = (loc.r**2).sum(dim=-1).sqrt().view(-1, 1)
        grad = loc.r/d
        grad = cat([grad, -grad])
        indices = cat([loc.j, loc.i])
        return d, grad, indices


def example():
    PairDistance = DummyPairDistance
    setup = Setup(3.0, [PairDistance(10, 10), PairDistance(10, 18),
                        PairDistance(18, 18)])
    xyz = np.stack(np.meshgrid([0, 1.5], [0, 1.5], [0, 1.5])
                   ).reshape(3, -1).transpose()
    numbers = 4*[10] + 4*[18]
    atoms = TorchAtoms(setup, positions=xyz, numbers=numbers)
    j0 = atoms.desc_0_indices
    j1 = atoms.desc_1_indices
    j2 = atoms.desc_2_indices


if __name__ == '__main__':
    example()

