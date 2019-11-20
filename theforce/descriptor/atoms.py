#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from torch import ones_like, as_tensor, from_numpy, cat
from ase.atoms import Atoms
from ase.neighborlist import NeighborList
from ase.calculators.singlepoint import SinglePointCalculator
import copy
import warnings
from theforce.util.util import iterable
from theforce.util.parallel import balance_work
import random
import itertools


def lex3(x):
    if x[0] != 0:
        return x[0] > -x[0]
    elif x[1] != 0:
        return x[1] > -x[1]
    elif x[2] != 0:
        return x[2] > -x[2]
    else:
        return True  # True or False here shouldn't make any difference


class Local:

    def __init__(self, i, j, a, b, r, off=None, descriptors=[], dont_save_grads=False):
        """
        i, j: indices
        a, b: atomic numbers
        r : r[j] - r[i]
        off: offsets
        """
        self._i = from_numpy(np.full_like(j, i))
        self._j = from_numpy(j)
        self._a = from_numpy(np.full_like(b, a))
        self._b = from_numpy(b)
        self._r = r
        self._m = ones_like(self._i).to(torch.bool)
        self.off = off
        self.stage(descriptors, dont_save_grads=dont_save_grads)

    def stage(self, descriptors, dont_save_grads=False):
        for desc in descriptors:
            desc.precalculate(self, dont_save_grads=dont_save_grads)

    @property
    def loc(self):
        """This is defined as a property in order to avoid memory leak."""
        return self

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
    def _lex(self):
        """This is defined as a property in order to avoid memory leak."""
        if self.off is None:
            return ones_like(self._i).to(torch.bool)
        else:
            return torch.tensor([lex3(a) for a in self.off], dtype=torch.bool)

    @property
    def lex(self):
        return self._lex[self._m]

    def select(self, a, b, bothways=False, in_place=True):
        m = (self._a == a) & (self._b == b)
        if a == b:
            if bothways:
                pass
            else:
                m = m & ((self._j > self._i) | ((self._j == self._i) &
                                                self._lex))
        elif a != b:
            if bothways:
                m = (m | ((self._a == b) & (self._b == a)))
            else:
                pass
        if in_place:
            self._m = m.to(torch.bool)
        return m.to(torch.bool)

    def unselect(self):
        self._m = ones_like(self._i).to(torch.bool)

    def as_atoms(self):
        a = self._a.unique().detach().numpy()
        atoms = (TorchAtoms(numbers=a, positions=len(a)*[(0, 0, 0)]) +
                 TorchAtoms(numbers=self._b.detach().numpy(), positions=self._r.detach().numpy()))
        if 'target_energy' in self.__dict__:
            atoms.set_calculator(
                SinglePointCalculator(atoms, energy=self.target_energy))
        return atoms

    def detach(self, keepids=False):
        a = self._a.detach().numpy()
        b = self._b.detach().numpy()
        r = self._r.clone()
        if keepids:
            i = self._i.detach().numpy()
            j = self._j.detach().numpy()
        else:
            i = np.zeros(a.shape[0], dtype=np.int)
            j = np.arange(1, a.shape[0]+1, dtype=np.int)
        return Local(i, j, a, b, r)

    def __eq__(self, other):
        if other.__class__ == TorchAtoms:
            return False
        elif (self._i != other._i).any():
            return False
        elif (self._j != other._j).any():
            return False
        elif (self._a != other._a).any():
            return False
        elif (self._b != other._b).any():
            return False
        elif (self._r != other._r).any():
            return False
        elif (self._lex != other._lex).any():
            return False
        else:
            return True


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
                 descriptors=[], group=None, **kwargs):
        super().__init__(**kwargs)

        if ase_atoms:
            self.__dict__ = ase_atoms.__dict__

        # ------------------------------- ----------
        if group is not None:
            self.attach_process_group(group)
        else:
            self.is_distributed = False
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
            if ase_atoms is not None and ase_atoms.get_calculator() is not None:
                if 'energy' in ase_atoms.calc.results:
                    self.target_energy = as_tensor(
                        ase_atoms.get_potential_energy())
                if 'forces' in ase_atoms.calc.results:
                    self.target_forces = as_tensor(ase_atoms.get_forces())

    def set_targets(self):
        self.target_energy = as_tensor(self.get_potential_energy())
        self.target_forces = as_tensor(self.get_forces())

    def attach_process_group(self, group):
        self.process_group = group
        self.is_distributed = True

    def detach_process_group(self):
        del self.process_group
        self.is_distributed = False

    def build_nl(self, rc):
        self.nl = NeighborList(self.natoms * [rc / 2], skin=0.0,
                               self_interaction=False, bothways=True)
        self.cutoff = rc
        self.xyz = torch.from_numpy(self.positions)
        try:
            self.lll = torch.from_numpy(self.cell)
        except TypeError:
            self.lll = torch.from_numpy(self.cell.array)
        # distributed setup
        if self.is_distributed:
            workers = torch.distributed.get_world_size(
                group=self.process_group)
            indices = balance_work(self.natoms, workers)
            rank = torch.distributed.get_rank(group=self.process_group)
            self.indices = range(*indices[rank])
        else:
            self.indices = range(self.natoms)

    def update(self, cutoff=None, descriptors=None, forced=False,
               posgrad=False, cellgrad=False, dont_save_grads=False):
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
            self.xyz.requires_grad = posgrad
            self.lll.requires_grad = cellgrad
            for a in self.indices:
                n, off = self.nl.get_neighbors(a)
                cells = (from_numpy(off[..., None].astype(np.float)) *
                         self.lll).sum(dim=1)
                r = self.xyz[n] - self.xyz[a] + cells
                self.loc += [Local(a, n, types[a], types[n],
                                   r, off, self.descriptors,
                                   dont_save_grads=dont_save_grads)]
            for loc in self.loc:
                loc.natoms = self.natoms

            self.changes.update_references()

    @property
    def natoms(self):
        return self.get_number_of_atoms()

    @property
    def tnumbers(self):
        return torch.from_numpy(self.numbers)

    @property
    def tpbc(self):
        return torch.from_numpy(self.pbc)

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

    def first_of_each_atom_type(atoms):
        indices = []
        unique = []
        for i, a in enumerate(atoms.numbers):
            if a not in unique:
                unique += [a]
                indices += [i]
        return indices

    def __eq__(self, other):  # Note: descriptors are excluded
        if other.__class__ == Local:
            return False
        elif self.natoms != other.natoms:
            return False
        elif (self.pbc != other.pbc).any():
            return False
        elif not self.lll.allclose(other.lll):
            return False
        elif (self.numbers != other.numbers).any():
            return False
        elif not self.xyz.allclose(other.xyz):
            return False
        else:
            return True

    def copy(self):
        new = TorchAtoms(positions=self.positions.copy(),
                         cell=self.cell.copy(),
                         numbers=self.numbers.copy(),
                         pbc=self.pbc.copy(),
                         descriptors=self.descriptors,
                         cutoff=self.cutoff,
                         group=self.process_group if self.is_distributed else None)
        vel = self.get_velocities()
        if vel is not None:
            new.set_velocities(vel)
        return new

    def set_cell(self, *args, **kwargs):
        super().set_cell(*args, **kwargs)
        try:
            self.lll = torch.from_numpy(self.cell)
        except TypeError:
            self.lll = torch.from_numpy(self.cell.array)

    def set_positions(self, *args, **kwargs):
        super().set_positions(*args, **kwargs)
        self.xyz = torch.from_numpy(self.positions)

    def as_ase(self):
        atoms = Atoms(positions=self.positions, cell=self.cell,
                      pbc=self.pbc, numbers=self.numbers)  # TODO: e, f
        vel = self.get_velocities()
        if vel is not None:
            atoms.set_velocities(vel)
        return atoms

    def as_local(self):
        """ As the inverse of Local.as_atoms """
        # positions[0] should to be [0, 0, 0]
        r = torch.as_tensor(self.positions[1:])
        a, b = np.broadcast_arrays(self.numbers[0], self.numbers[1:])
        _i = np.arange(self.natoms)
        i, j = np.broadcast_arrays(_i[0], _i[1:])
        loc = Local(i, j, a, b, r)
        if 'target_energy' in self.__dict__:
            loc.target_energy = self.target_energy
        return loc

    def shake(self, beta=0.05, update=True):
        trans = np.random.laplace(0., beta, size=self.positions.shape)
        self.translate(trans)
        if update:
            self.update()

    def single_point(self):
        results = {}
        for q in ['energy', 'forces', 'stress', 'xx']:
            try:
                results[q] = self.calc.results[q]
            except KeyError:
                pass
        self.set_calculator(SinglePointCalculator(self, **results))


class AtomsData:

    def __init__(self, X=None, traj=None, posgrad=False, cellgrad=False, **kwargs):
        if X is not None:
            self.X = X
            assert self.check_content()
        elif traj is not None:
            self.X = []
            from ase.io import Trajectory
            t = Trajectory(traj, 'r')
            self.X += [TorchAtoms(ase_atoms=atoms, **kwargs)
                       for atoms in t]
            t.close()
        else:
            raise RuntimeError('AtomsData invoked without any input')
        self.posgrad = posgrad
        self.cellgrad = cellgrad

    def check_content(self):
        return all([atoms.__class__ == TorchAtoms for atoms in self])

    def numbers_set(self):
        _num = set()
        for atoms in self:
            for n in set(atoms.numbers):
                _num.add(n)
        numbers = sorted(list(_num))
        return numbers

    def pairs_set(self, numbers=None):
        if numbers is None:
            numbers = self.numbers_set()
        pairs = ([(a, b) for a, b in itertools.combinations(numbers, 2)] +
                 [(a, a) for a in numbers])
        return pairs

    def apply(self, operation, *args, **kwargs):
        for atoms in self.X:
            getattr(atoms, operation)(*args, **kwargs)

    def set_gpp(self, gpp, cutoff=None):
        self.apply('update', cutoff=cutoff,
                   descriptors=gpp.kern.kernels, forced=True)

    def update(self, *args, **kwargs):
        for atoms in self.X:
            atoms.update(*args, **kwargs)

    def update_nl_if_requires_grad(self, descriptors=None, forced=False):
        if self.trainable:
            for atoms in self.X:
                atoms.update(descriptors=descriptors, forced=forced,
                             posgrad=self.posgrad, cellgrad=self.cellgrad)

    def set_per_atoms(self, quant, values):
        vals = torch.split(values, split_size_or_sections=1)
        for atoms, v in zip(*[self, vals]):
            setattr(atoms, quant, v)

    def set_per_atom(self, quant, values):
        vals = torch.split(values, split_size_or_sections=self.natoms)
        for atoms, v in zip(*[self, vals]):
            setattr(atoms, quant, v)

    def shake(self, **kwargs):
        for atoms in self.X:
            atoms.shake(**kwargs)

    @property
    def natoms(self):
        return [atoms.natoms for atoms in self]

    @property
    def params(self):
        return [atoms.xyz for atoms in self.X if atoms.xyz.requires_grad]

    @property
    def trainable(self):
        return self.posgrad or self.cellgrad

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
        if isinstance(k, int):
            return self.X[k]
        else:
            return AtomsData(self.X[k])

    def __len__(self):
        return len(self.X)

    def to_traj(self, trajname, mode='w'):
        from ase.io import Trajectory
        t = Trajectory(trajname, mode)
        for atoms in self:
            t.write(atoms)
        t.close()

    def pick_random(self, n):
        if n > len(self):
            warnings.warn('n > len(AtomsData) in pick_random')
        return AtomsData(X=[self[k] for k in torch.randperm(len(self))[:n]])

    def append(self, others):
        if id(self) == id(others):
            _others = others.X[:]
        else:
            _others = iterable(others, ignore=TorchAtoms)
        for atoms in _others:
            assert atoms.__class__ == TorchAtoms
            self.X += [atoms]

    def __add__(self, other):
        if other.__class__ == AtomsData:
            return AtomsData(X=self.X+other.X)
        else:
            raise NotImplementedError(
                'AtomsData + {} is not implemented'.format(other.__class__))

    def __iadd__(self, others):
        self.append(others)
        return self

    def to_locals(self, keepids=False):
        return LocalsData([loc.detach(keepids=keepids) for atoms in self for loc in atoms])

    def sample_locals(self, size, keepids=False):
        return LocalsData([random.choice(random.choice(self)).detach(keepids=keepids)
                           for _ in range(size)])


class LocalsData:

    def __init__(self, X=None, traj=None):
        if X is not None:
            self.X = []
            for loc in X:
                assert loc.__class__ == Local
                self.X += [loc]
        elif traj is not None:
            from ase.io import Trajectory
            t = Trajectory(traj, 'r')
            self.X = []
            for atoms in t:
                tatoms = TorchAtoms(ase_atoms=atoms)
                if not np.allclose(tatoms.positions[0], np.zeros((3,))):
                    raise RuntimeError
                self.X += [tatoms.as_local()]
            t.close()
        else:
            raise RuntimeError('LocalsData invoked without any input')
        self.trainable = False

    def stage(self, descriptors, dont_save_grads=True):
        for loc in self:
            loc.stage(descriptors, dont_save_grads=dont_save_grads)

    def to_traj(self, trajname, mode='w'):
        from ase.io import Trajectory
        t = Trajectory(trajname, mode)
        for loc in self:
            t.write(loc.as_atoms())
        t.close()

    def __iter__(self):
        for locs in self.X:
            yield locs

    def __getitem__(self, k):
        if isinstance(k, int):
            return self.X[k]
        else:
            return LocalsData(self.X[k])

    def __len__(self):
        return len(self.X)

    def append(self, others, detach=False):
        if id(self) == id(others):
            _others = others.X[:]
        else:
            _others = iterable(others)
        for loc in _others:
            assert loc.__class__ == Local
            self.X += [loc.detach() if detach else loc]

    def __add__(self, other):
        if other.__class__ == LocalsData:
            return LocalsData(X=self.X+other.X)
        else:
            raise NotImplementedError(
                'AtomsData + {} is not implemented'.format(other.__class__))

    def __iadd__(self, others):
        self.append(others)
        return self


def sample_atoms(file, size=-1, chp=None, indices=None):
    """
    If 
        A = sample_atomsdata('atoms.traj', size=n, chp='data.chp')
        B = sample_atomsdata('data.chp')
    then,
        B = A
    """
    from ase.io import Trajectory

    # from traj
    if file.endswith('.traj'):
        traj = Trajectory(file)
        if size > len(traj):
            warnings.warn('size > len({})'.format(file))
        if indices is None:
            indices = np.random.permutation(len(traj))[:size].tolist()
        if chp:
            with open(chp, 'w') as ch:
                ch.write(file+'\n')
                for k in indices:
                    ch.write('{} '.format(k))
        return AtomsData(X=[TorchAtoms(ase_atoms=traj[k]) for k in indices])

    # from checkpoint
    elif file.endswith('.chp'):
        with open(file, 'r') as ch:
            _file = ch.readline().strip()
            _indices = [int(i) for i in ch.readline().split()]
        return sample_atoms(_file, indices=_indices)

    # other
    else:
        raise NotImplementedError('format {} is not recognized'.format(file))


def diatomic(numbers, distances, pbc=False, cell=None):
    from theforce.util.util import iterable
    from itertools import combinations
    if not hasattr(numbers[0], '__iter__'):
        nums = ([(a, b) for a, b in combinations(set(numbers), 2)] +
                [(a, a) for a in set(numbers)])
    else:
        nums = numbers
    X = [TorchAtoms(positions=[[0., 0., 0.], [d, 0., 0.]], numbers=n, cell=cell, pbc=pbc)
         for n in nums for d in iterable(distances)]
    if len(X) > 1:
        return AtomsData(X=X)
    else:
        return X[0]


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

    other = atoms.copy()
    print(other == atoms)

    for loc in atoms:
        print(loc.as_atoms().as_local() == loc.detach())

    empty = TorchAtoms(positions=[(0, 0, 0)], cutoff=3.)
    empty[0].detach()._r


if __name__ == '__main__':
    example()

