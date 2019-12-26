#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from ase.io import read, Trajectory
from theforce.util.util import iterable


class TrajAnalyser:

    def __init__(self, traj):
        self.traj = Trajectory(traj)
        self.numbers = self.traj[0].get_atomic_numbers()
        self.masses = self.traj[0].get_masses()

    @property
    def last(self):
        return self.traj.__len__()

    def select(self, *args):
        if len(args) == 0:
            return np.zeros_like(self.numbers)
        elif 'all' in args:
            return np.ones_like(self.numbers)
        else:
            return np.stack([self.numbers == a for a in iterable(args)]).any(axis=0)

    def get_pair(self, i, j):
        return self.traj[i], self.traj[j]

    def get_slice(self, start=0, stop=-1, step=1):
        if stop == -1:
            stop = self.last
        return start, stop, step

    def slice(self, **kwargs):
        for i in range(*self.get_slice(**kwargs)):
            yield self.traj[i]

    def center_of_mass(self, select='all', prop=('positions',), **kwargs):
        I = self.select(select)
        data = []
        for atoms in self.slice(**kwargs):
            data += [[getattr(atoms, f'get_{q}')()[I].sum(axis=0)
                      for q in prop]]
        return zip(*data)

    def displacements(self, deltas, numbers='all', sample_size=100, step_range=None, corr=None, stats=None):
        I = self.select(numbers)
        s = Sampler(*step_range) if step_range else Sampler(0, self.last)
        if corr is None:
            corr = corrlator
        if stats is None:
            stats = mean_var
        data = [[stats(data) for data in zip(*[iterable(corr(*self.get_pair(*s.sample_pair(delta)), I))
                                               for _ in range(sample_size)])] for delta in deltas]
        results = [list(zip(*[dat[j] for dat in data]))
                   for j in range(len(data[0]))]
        return results


class Sampler:

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def sample(self):
        return np.random.randint(self.start, self.stop)

    def sample_pair(self, delta):
        i = np.random.randint(self.start, self.stop-delta)
        return i, i+delta


def corrlator(a, b, I):
    r = b.get_positions()[I]-a.get_positions()[I]
    msd = (r**2).sum(axis=-1).mean()
    smd = (r.sum(axis=0)**2).sum()/r.shape[0]
    return msd, smd


def mean_var(data):
    return np.mean(data), np.sqrt(np.var(data))


def mean_squared_displacement(traj, start=0, stop=-1, step=1, origin=None, numbers=None):
    if not origin:
        origin = start
    atoms = read(traj, origin)
    xyz0 = atoms.get_positions()
    if numbers is None:
        numbers = np.unique(atoms.numbers)
    sl = '{}:{}:{}'.format(start, stop, step)
    xyz = np.stack([atoms.get_positions() for atoms in read(traj, sl)])
    D = ((xyz - xyz0)**2).sum(axis=-1)
    msd = [(D[:, atoms.numbers == number]).mean(axis=1) for number in numbers]
    return numbers, msd

