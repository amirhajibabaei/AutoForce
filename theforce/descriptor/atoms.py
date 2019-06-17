
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

    def __init__(self, i, j, a, b, r, im, descriptors=[]):
        """
        i, j: indices
        a, b: atomic numbers
        r : r[j] - r[i]
        im: image
        """
        self._i = from_numpy(np.full_like(j, i))
        self._j = from_numpy(j)
        self._a = from_numpy(np.full_like(b, a))
        self._b = from_numpy(b)
        self._r = r
        self._m = ones_like(self._i).to(torch.bool)
        self._im = from_numpy(im).byte()
        self.loc = self
        self.stage(descriptors)

    def stage(self, descriptors):
        for desc in descriptors:
            desc.calculate(self)

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

    @property
    def image(self):
        return self._im[self._m]

    def select(self, a, b, bothways=False, in_place=True):
        m = (self._a == a) & (self._b == b)
        if bothways and a != b:
            m = (m | ((self._a == b) & (self._b == a)))
        if not bothways and a == b:
            m = (m & (self._im | (self._j > self._i)))
        if in_place:
            self._m = m.to(torch.bool)
        return m.to(torch.bool)

    def unselect(self):
        self._m = ones_like(self._i).to(torch.bool)


class AtomsChanges:

    def __init__(self, atoms):
        self._ref = atoms
        self.update_references()

    def update_references(self):
        self._natoms = self._ref.natoms
        self._numbers = self._ref.numbers.copy()
        self._positions = self._ref.positions.copy()
        self._cell = self._ref.cell.copy()
        self._pbc = self._ref.pbc.copy()
        self._descriptors = [kern.state for kern in
                             self._ref.descriptors]

    @property
    def natoms(self):
        return self._ref.natoms != self._natoms

    @property
    def atomic_numbers(self):
        return (self._ref.numbers != self._numbers).any()

    @property
    def numbers(self):
        return (self.natoms or self.atomic_numbers)

    @property
    def positions(self):
        return not np.allclose(self._ref.positions, self._positions)

    @property
    def cell(self):
        return not np.allclose(self._ref.cell, self._cell)

    @property
    def pbc(self):
        return (self._ref.pbc != self._pbc).any()

    @property
    def atoms(self):
        return any([self.numbers, self.positions, self.cell, self.pbc])

    @property
    def descriptors(self):
        return [c != r.state for c, r in zip(*[self._descriptors, self._ref.descriptors])]


class TorchAtoms(Atoms):

    def __init__(self, ase_atoms=None, energy=None, forces=None, cutoff=None,
                 descriptors=[], **kwargs):
        super().__init__(**kwargs)

        if ase_atoms:
            self.__dict__ = ase_atoms.__dict__

        # ------------------------------- ----------
        self.cutoff = cutoff
        self.descriptors = descriptors
        self.changes = AtomsChanges(self)
        if cutoff is not None:
            self.build_nl(cutoff)
            self.update(forced=True)
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

    def build_nl(self, rc):
        self.nl = NeighborList(self.natoms * [rc / 2], skin=0.0,
                               self_interaction=False, bothways=True)
        self.cutoff = rc

    def update(self, cutoff=None, descriptors=None, forced=False,
               posgrad=False, cellgrad=False):
        if cutoff or self.changes.numbers:
            self.build_nl(cutoff if cutoff else self.cutoff)
            forced = True
        if descriptors:
            self.descriptors = descriptors
            forced = True
        if forced or self.changes.atoms:
            self.nl.update(self)
            self.loc = []
            types = self.get_atomic_numbers()
            xyz = torch.from_numpy(self.positions)
            xyz.requires_grad = posgrad
            cell = torch.from_numpy(self.cell)
            cell.requires_grad = cellgrad
            for a in range(self.natoms):
                n, off = self.nl.get_neighbors(a)
                image = (off != 0).any(axis=1)
                cells = (from_numpy(off[..., None].astype(np.float)) *
                         cell).sum(dim=1)
                r = xyz[n] - xyz[a] + cells
                self.loc += [Local(a, n, types[a], types[n],
                                   r, image, self.descriptors)]
            for loc in self.loc:
                loc.natoms = self.natoms

            self.changes.update_references()
            self.xyz = xyz
            self.lll = cell

    @property
    def natoms(self):
        return self.get_number_of_atoms()

    def __getattr__(self, attr):
        try:
            return torch.cat([env.__dict__[attr] for env in self.loc])
        except KeyError:
            raise AttributeError()

    def __getitem__(self, k):
        """This is a overloads the behavior of ase.Atoms."""
        return self.loc[k]

    def __iter__(self):
        """This is a overloads the behavior of ase.Atoms."""
        for env in self.loc:
            yield env

    def copy(self):
        new = TorchAtoms(positions=self.positions.copy(),
                         cell=self.cell.copy(),
                         numbers=self.numbers.copy(),
                         pbc=self.pbc.copy(),
                         descriptors=self.descriptors,
                         cutoff=self.cutoff)
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
    atoms.copy()


if __name__ == '__main__':
    example()

