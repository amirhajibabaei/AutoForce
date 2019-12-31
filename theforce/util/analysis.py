#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from ase.io import read, Trajectory
from theforce.util.util import iterable
from scipy.stats import bayes_mvs as stats


class TrajAnalyser:

    def __init__(self, traj, start=0, stop=-1):
        self.traj = Trajectory(traj)
        self.numbers = self.traj[0].get_atomic_numbers()
        self.masses = self.traj[0].get_masses()
        self.set_range(start, stop)

    def set_range(self, start, stop):
        self._start = start
        self._stop = stop

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop if self._stop >= 0 else self.last-self._stop

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

    def get_rand_pair(self, s, delta):
        succ = False
        while not succ:
            try:
                i, j = s.sample_pair(delta)
                a, b = self.traj[i], self.traj[j]
                succ = True
            except:
                pass
        return a, b

    def get_slice(self, start=None, stop=None, step=1):
        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop
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

    def ave_vol(self, srange=None, sample_size=100, stats=None):
        s = Sampler(*srange) if srange else Sampler(self.start, self.stop)
        if stats is None:
            stats = mean_var
        v = stats([self.traj[s.sample()].get_volume()
                   for _ in range(sample_size)])
        return v

    def displacements(self, numbers='all', deltas=None, srange=None, sample_size=100, corr=None, stats=None):
        I = self.select(numbers)
        s = Sampler(*srange) if srange else Sampler(self.start, self.stop)
        if deltas is None:
            deltas = get_exponential_deltas(s.start, s.stop)
        if corr is None:
            corr = correlator
        if stats is None:
            stats = mean_var
        data = [[stats(data) for data in zip(*[iterable(corr(*self.get_rand_pair(s, delta), I))
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


def get_exponential_deltas(start, stop, n=6):
    i = stop-start
    j = 1
    k = 0
    while j < i:
        j *= 2
        k += 1
    deltas = [2**(k-j-2) for j in range(0, n)][::-1]
    return [t for t in deltas if t > 1]


def correlator(a, b, I):
    r = b.get_positions()[I]-a.get_positions()[I]
    msd = (r**2).sum(axis=-1).mean()
    smd = (r.sum(axis=0)**2).sum()/r.shape[0]
    return msd, smd


def mean_var(data):
    return np.mean(data), np.var(data)


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

