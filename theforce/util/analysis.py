#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from ase.io import read, Trajectory
from theforce.util.util import iterable
from scipy.stats import bayes_mvs as stats


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

    def displacements(self, numbers='all', deltas=None, step_range=None, sample_size=100, corr=None, stats=None):
        I = self.select(numbers)
        s = Sampler(*step_range) if step_range else Sampler(0, self.last)
        if deltas is None:
            deltas = get_exponential_deltas(s.start, s.stop)
        if corr is None:
            corr = corrlator
        if stats is None:
            stats = mean_var
        data = [[stats(data) for data in zip(*[iterable(corr(*self.get_pair(*s.sample_pair(delta)), I))
                                               for _ in range(sample_size)])] for delta in deltas]
        results = [list(zip(*[dat[j] for dat in data]))
                   for j in range(len(data[0]))]
        return deltas, results

    def diffusion_constants(self, dt=1., numbers='all', sample_size=100):
        deltas, results = self.displacements(
            numbers=numbers, sample_size=sample_size, stats=stats)
        time = np.array(deltas)*dt
        msd = np.array([d.statistic for d in results[0][0]])
        msd_err = np.array([d.statistic for d in results[0][2]])
        smd = np.array([d.statistic for d in results[1][0]])
        smd_err = np.array([d.statistic for d in results[1][2]])
        Dt = tuple(a/6 for a in get_slopes(time, msd, msd_err))
        Dj = tuple(a/6 for a in get_slopes(time, smd, smd_err))
        return Dt, Dj


class Sampler:

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def sample(self):
        return np.random.randint(self.start, self.stop)

    def sample_pair(self, delta):
        i = np.random.randint(self.start, self.stop-delta)
        return i, i+delta


def get_exponential_deltas(start, end):
    i = end-start
    j = 1
    n = 0
    while j < i:
        j *= 2
        n += 1
    deltas = [2**(n-j-2) for j in range(0, 6)][::-1]
    return deltas


def corrlator(a, b, I):
    r = b.get_positions()[I]-a.get_positions()[I]
    msd = (r**2).sum(axis=-1).mean()
    smd = (r.sum(axis=0)**2).sum()/r.shape[0]
    return msd, smd


def mean_var(data):
    return np.mean(data), np.sqrt(np.var(data))


def get_slopes(x, y, yerr):
    a = np.polyfit(x, y, 1)
    b = np.polyfit(x, y-yerr, 1)
    c = np.polyfit(x, y+yerr, 1)
    return a[0], b[0], c[0]


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

