
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from torch import ones_like, as_tensor, from_numpy, cat
from ase.atoms import Atoms
from ase.neighborlist import NeighborList
from ase.calculators.calculator import PropertyNotImplementedError
import copy


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
            desc.calculate(self)
        self.loc = self

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


class LocalEnvirons(NeighborList):

    def __init__(self, atoms, rc):
        """
        This can be used instead of Neighborlist in calculators.
        """
        self.atoms = atoms
        self.rc = rc
        cutoffs = atoms.natoms*[rc / 2]
        super().__init__(cutoffs, skin=0.0, self_interaction=False, bothways=True)
        self._copy = np.zeros_like(atoms.positions)

    def update(self, descriptors=[], forced=False):
        if forced or not np.allclose(self.atoms.positions, self._copy):
            self._copy[:] = self.atoms.positions[:]
            super().update(self.atoms)
            self.loc = []
            types = self.atoms.get_atomic_numbers()
            for a in range(self.atoms.natoms):
                n, off = self.nl.get_neighbors(a)
                cells = from_numpy(np.dot(off, self.atoms.cell))
                r = self.atoms.xyz[n] - self.atoms.xyz[a] + cells
                self.loc += [Local(a, n, types[a], types[n],
                                   r, descriptors)]
            for loc in self.loc:
                loc.natoms = self.atoms.natoms

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
        try:
            return torch.cat([getattr(env, attr) for env in self.__dict__['loc']])
        except KeyError:
            raise AttributeError('in LocalEnvirons')


class TorchAtoms(Atoms):

    def __init__(self, ase_atoms=None, energy=None, forces=None, cutoff=None,
                 descriptors=[], grad=False, **kwargs):
        """xyz, loc, descriptors added to Atoms object."""
        super().__init__(**kwargs)

        if ase_atoms:
            self.__dict__ = ase_atoms.__dict__

        #------------------------------- gradly-tensors
        self.xyz = from_numpy(self.positions)
        self.xyz.requires_grad = grad
        self.descriptors = descriptors
        if cutoff is not None:
            self.build_loc(cutoff)
            self.update()
        else:
            self.loc = None
        # ------------------------------------------

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

    def build_loc(self, rc):
        self.loc = LocalEnvirons(self, rc)

    def update(self, cutoff=None, descriptors=None, forced=False):
        if cutoff is not None:
            self.build_loc(cutoff)
        if descriptors is not None:
            self.descriptors = descriptors
        self.loc.update(descriptors=self.descriptors, forced=forced)

    @property
    def natoms(self):
        return self.get_number_of_atoms()

    def __getattr__(self, attr):
        try:
            return getattr(self.__dict__['loc'], attr)
        except KeyError:
            raise AttributeError('in TorchAtoms')

    def __getitem__(self, k):
        """This is a overloads the behavior of ase.Atoms."""
        return self.loc[k]

    def __iter__(self):
        """This is a overloads the behavior of ase.Atoms."""
        for env in self.loc:
            yield env

    def copy(self):
        xyz = self.xyz
        loc = self.loc
        descriptors = self.descriptors
        del self.xyz, self.loc, self.descriptors
        new = copy.deepcopy(self)
        new.xyz = torch.from_numpy(new.positions)
        new.xyz.requires_grad = xyz.requires_grad
        new.descriptors = descriptors
        try:
            new.build_loc(loc.rc)
            new.update()
        except AttributeError:
            pass
        self.xyz = xyz
        self.loc = loc
        self.descriptors = descriptors
        return new


class AtomsData:

    def __init__(self, X=[], traj=None, **kwargs):
        self.X = X
        if traj:
            from ase.io import Trajectory
            self.X += [TorchAtoms(ase_atoms=atoms, **kwargs)
                       for atoms in Trajectory(traj)]

    def apply(self, operation, *args, **kwargs):
        for atoms in self.X:
            getattr(atoms, operation)(*args, **kwargs)

    def set_gpp(self, gpp, cutoff=None):
        self.apply('update', cutoff=cutoff,
                   descriptors=gpp.kern.kernels, forced=True)

    def update_nl_if_requires_grad(self, descriptors=[], forced=True):
        for atoms in self.X:
            if atoms.xyz.requires_grad:
                atoms.update(descriptors=descriptors, forced=forced)

    def set_per_atoms(self, quant, values):
        vals = torch.split(values, split_size_or_sections=1)
        for atoms, v in zip(*[self, vals]):
            setattr(atoms, quant, v)

    def set_per_atom(self, quant, values):
        vals = torch.split(values, split_size_or_sections=self.natoms)
        for atoms, v in zip(*[self, vals]):
            setattr(atoms, quant, v)

    @property
    def natoms(self):
        return [atoms.natoms for atoms in self]

    @property
    def params(self):
        return [atoms.xyz for atoms in self.X if atoms.xyz.requires_grad]

    @property
    def target_energy(self):
        return torch.tensor([atoms.target_energy for atoms in self])

    @property
    def target_forces(self):
        return torch.cat([atoms.target_forces for atoms in self])

    def cat(self, attr):
        return torch.cat([getattr(atoms, attr) for atoms in self])

    def __iter__(self):
        for atoms in self.X:
            yield atoms

    def __getitem__(self, k):
        return self.X[k]

    def __len__(self):
        return len(self.X)


def namethem(descriptors, base='D'):
    for i, desc in enumerate(descriptors):
        desc.name = base+'_{}'.format(i)


def example():
    from theforce.similarity.pair import DistanceKernel
    from theforce.regression.core import SquaredExp
    kerns = [DistanceKernel(SquaredExp(), 10, 10),
             DistanceKernel(SquaredExp(), 10, 18),
             DistanceKernel(SquaredExp(), 18, 18)]
    namethem(kerns)
    xyz = np.stack(np.meshgrid([0, 1.5], [0, 1.5], [0, 1.5])
                   ).reshape(3, -1).transpose()
    numbers = 4*[10] + 4*[18]
    atoms = TorchAtoms(positions=xyz, numbers=numbers,
                       cutoff=3.0, descriptors=kerns)
    atoms.update(descriptors=kerns, forced=True)

    # copy
    b = atoms.copy()
    atoms.xyz.requires_grad = True
    c = atoms.copy()


if __name__ == '__main__':
    example()

