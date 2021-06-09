# +
from theforce.analysis.kde import Gaussian_kde
from theforce.math.ql import Ql
import torch
import numpy as np
from ase.units import kB


class Meta:

    def __init__(self, colvar, sigma=0.1, w=0.01, tem=None):
        """
        colvar: a function which returns the CVs
        sigma:  the band-width for deposited Gaussians (Ang)
        w:      the height of the Gaussians multiplied by dt (eV*fs)
        tem:    if given -> well-tempered metadynamics (K)
        ---------------------------------------------
        example for colvar:
        def colvar(numbers, xyz, cell, pbc, nl):
            return (xyz[1]-xyz[0]).norm()
        """
        self.colvar = colvar
        self.kde = Gaussian_kde(sigma)
        self.w = w
        self.tem = tem
        with open('meta.hist', 'w') as hst:
            hst.write(f'# {sigma}\n')

    def __call__(self, calc):
        kwargs = {'op': '+=', 'is_meta': True}
        self.rank = calc.rank
        if calc.rank == 0:
            cv = self.colvar(calc.atoms.numbers,
                             calc.atoms.xyz,
                             calc.atoms.lll,
                             calc.atoms.pbc,
                             calc.atoms.nl)
            energy = self.energy(cv)
            self._cv = cv.detach()
        else:
            energy = torch.zeros(1)
        return energy, kwargs

    def energy(self, cv):
        kde = self.kde(cv, density=False)
        energy = self.w*kde
        if self.tem is not None:
            gamma = 1./(kB*self.tem)
            energy = (1 + energy*gamma).log()/gamma
        return energy

    def update(self):
        if self.rank == 0:
            self.kde.count(self._cv)
            with open('meta.hist', 'a') as hst:
                for f in self._cv:
                    hst.write(f' {float(f)}')
                hst.write('\n')


class Posvar:

    def __init__(self, index, select=None):
        self.index = index
        self.select = select

    def __call__(self, numbers, xyz, cell, pbc, nl):
        a = torch.ones(len(numbers), 1)
        a[self.index] = 0.
        if self.select is None:
            p = xyz
        else:
            select = numbers == self.select
            a = a[select]
            p = xyz[select]
        c = xyz[self.index] - (a*p).mean(dim=0)
        return c


class Qlvar:

    def __init__(self, i, j, index=None, cutoff=4., l=[6]):
        """
        i:      type of the atom (Z) for which ql will be calculated
        j:      type of the atoms (Z) in the environment which contribute to ql
        index:  index of the atom for which ql will be calculated, if None -> the first found
        cutoff: cutoff for atoms in the neighborhood of i
        l:      angular indices for ql
        """
        self.i = i
        self.j = j
        self.index = index
        self.var = Ql(max(l), cutoff)
        self.l = l

    def __call__(self, numbers, xyz, cell, pbc, nl):
        if self.index is None:
            self.index = np.where(numbers == self.i)[0][0]
        i = self.index
        if numbers[i] != self.i:
            raise RuntimeError(f'numbers[{i}] != {self.i}')
        nei_i, off = nl.get_neighbors(i)
        off = torch.from_numpy(off).type(xyz.type())
        off = (off[..., None]*cell).sum(dim=1)
        env = numbers[nei_i] == self.j
        r_ij = xyz[nei_i[env]] - xyz[i] + off[env]
        ql = self.var(r_ij).index_select(0, torch.tensor(self.l))
        return ql


class Catvar:

    def __init__(self, *var):
        self.var = var

    def __call__(self, *args):
        return torch.cat([var(*args).view(-1) for var in self.var])
