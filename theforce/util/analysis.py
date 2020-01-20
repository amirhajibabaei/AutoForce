#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from ase.io import read, Trajectory
from ase.neighborlist import NeighborList, natural_cutoffs
from theforce.descriptor.atoms import AtomsData, TorchAtoms
from theforce.util.util import iterable
from theforce.util.rdf import rdf
from theforce.descriptor.sphcart import cart_coord_to_sph, sph_coord_to_cart
from scipy.stats import bayes_mvs as stats


def no_transform(atoms):
    return atoms


def standard_cell_transform(atoms):
    atoms.set_cell(atoms.cell.cellpar(), scale_atoms=True)
    return atoms


class TrajAnalyser:

    def __init__(self, traj, start=0, stop=-1, transform=no_transform):
        self.traj = Trajectory(traj)
        self.transform = transform
        self.numbers = self[0].get_atomic_numbers()
        self.masses = self[0].get_masses()
        self.set_range(start, stop)
        self.indices = np.arange(0, self.numbers.shape[0])

    def set_range(self, start, stop):
        self._start = start
        self._stop = stop

    def __getitem__(self, k):
        return self.transform(self.traj[k])

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self.last if self._stop == -1 else self._stop

    @property
    def last(self):
        return self.traj.__len__()

    def select(self, *args):
        if len(args) == 0:
            return np.full(self.numbers.shape[0], False)
        elif 'all' in args:
            return np.full(self.numbers.shape[0], True)
        else:
            return np.stack([self.numbers == a for b in iterable(args)
                             for a in iterable(b)]).any(axis=0)

    def select_indices(self, *args):
        return self.indices[self.select(*args)]

    def get_pair(self, i, j):
        return self[i], self[j]

    def get_rand_pair(self, s, delta):
        succ = False
        while not succ:
            try:
                i, j = s.sample_pair(delta)
                a, b = self[i], self[j]
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
            yield self[i]

    def get_scalars(self, prop=('volume',), **kwargs):
        data = []
        for atoms in self.slice(**kwargs):
            data += [[getattr(atoms, f'get_{q}')()
                      for q in prop]]
        return zip(*data)

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
        v = stats([self[s.sample()].get_volume()
                   for _ in range(sample_size)])
        return v

    def msd(self, select='all', wrt=None, **kwargs):
        I = self.select(select)
        start, stop, step = self.get_slice(**kwargs)
        if wrt is None:
            wrt = start
        x = self[wrt].get_positions()[I]
        d = np.array([((atoms.get_positions()[I]-x)**2).sum(axis=-1).mean()
                      for atoms in self.slice(**kwargs)])
        return np.arange(start, stop, step), d

    def displacements(self, select='all', deltas=None, srange=None, sample_size=100, corr=None, stats=None):
        I = self.select(select)
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

    def diffusion_constants(self, dt=1., select='all', sample_size=100):
        deltas, results = self.displacements(
            select=select, sample_size=sample_size, stats=stats)
        time = np.array(deltas)*dt
        msd = np.array([d.statistic for d in results[0][0]])
        msd_err = np.array([d.statistic for d in results[0][2]])
        smd = np.array([d.statistic for d in results[1][0]])
        smd_err = np.array([d.statistic for d in results[1][2]])
        Dt = tuple(a/6 for a in get_slopes(time, msd, msd_err))
        Dj = tuple(a/6 for a in get_slopes(time, smd, smd_err))
        return Dt, Dj

    def hist_rtp_displacements(self, delta, rmax=10., bins=[100, 30, 60], select='all',
                               srange=None, sample_size=100):
        """
        delta: steps
        returns: r, theta, phi, histogram, density (of selected atoms)
        """
        I = self.select(select)
        s = Sampler(*srange) if srange else Sampler(self.start, self.stop)
        _bins = [np.linspace(0, rmax, bins[0]),
                 np.linspace(0, np.pi, bins[1]),
                 np.linspace(-np.pi, np.pi, bins[2])]
        h = np.zeros(shape=np.array(bins)-1)
        vol = []
        for _ in range(sample_size):
            a, b = self.get_rand_pair(s, delta)
            vol += [a.get_volume(), b.get_volume()]
            d = (b.positions[I] - a.positions[I]).reshape(-1, 3)
            rtp = np.array(cart_coord_to_sph(*d.T)).T
            h += np.histogramdd(rtp, bins=_bins)[0]
        r, t, p = (a[:-1] + (a[1]-a[0])/2 for a in _bins)
        N = I.sum()
        h /= (N*sample_size)
        rho = N/np.array(vol).mean()
        return r, t, p, h, rho

    def get_positions(self, select='all', **kwargs):
        I = self.select(select)
        return np.stack([atoms.get_positions()[I] for atoms in self.slice(**kwargs)])

    def write_traj(self, out, **kwargs):
        traj = Trajectory(out, 'w')
        for atoms in self.slice(**kwargs):
            traj.write(atoms)

    def rdf(self, rmax, nbins, select='all', srange=None, sample_size=100, file=None):
        I = self.select(select)
        s = Sampler(*srange) if srange else Sampler(self.start, self.stop)
        data = AtomsData([TorchAtoms(self[s.sample()][I])
                          for _ in range(sample_size)])
        r, gdict = rdf(data, rmax, nbins)
        if file is not None:
            #header = 'r ' + ' '.join(f'{key}' for key in gdict.keys())
            header = ' '.join(f'{k[0]}-{k[1]}' for k in gdict.keys())
            out = np.stack([r, ]+[gdict[key] for key in gdict.keys()]).T
            np.savetxt(file, out, header=header)
        return r, gdict

    @staticmethod
    def read_rdf(file):
        g = np.loadtxt(file)
        gdict = {}
        with open(file) as f:
            header = f.readline()
        for a, b in zip(*[header.split(), g.T]):
            if a == '#':
                r = b
            else:
                key = tuple(int(v) for v in a.split('-'))
                gdict[key] = b
        return r, gdict

    def attach_nl(self, cutoff=None, si=False, bw=True):
        if cutoff is None:
            coff = natural_cutoffs(self[0])
        elif type(cutoff) is float:
            coff = np.full_like(self[0].numbers, cutoff)
        elif type(cutoff) is dict:
            coff = list(map(cutoff.get, self[0].numbers))
        elif type(cutoff) is list:
            coff = cutoff
        self.nl = NeighborList(
            coff, skin=0.0, self_interaction=si, bothways=bw)

    def get_neighbors(self, j, atoms=None, only=None):

        if atoms:
            if type(atoms) == int:
                atoms = self[atoms]
            self.nl.update(atoms)

        k, off = self.nl.get_neighbors(j)
        if only:
            I = self.numbers[k] == only
            k = k[I]
            off = off[I]
        return k, off

    def get_mic(self, atoms, j, k, off):
        cells = (off[..., None].astype(np.float) *
                 atoms.cell).sum(axis=1)
        r = atoms.positions[k] - atoms.positions[j] + cells
        return r


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

