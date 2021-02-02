# +
from theforce.util.kde import Gaussian_kde
from theforce.math.ql import Ql
import torch


class Meta:

    def __init__(self, colvar, sigma=0.1, w=0.01):
        """
        colvar: a function which returns the CVs
        sigma: the band-width for deposited Gaussians
        w: the height of the Gaussians
        ---------------------------------------------
        example for colvar:
        def colvar(numbers, xyz, cell, pbc, nl):
            return (xyz[1]-xyz[0]).norm()
        """
        self.colvar = colvar
        self.kde = Gaussian_kde(sigma)
        self.w = w
        with open('meta.hist', 'w') as hst:
            hst.write(f'# {sigma}\n')

    def __call__(self, calc):
        kwargs = {'op': '+='}
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
        return energy

    def update(self):
        if self.rank == 0:
            self.kde.count(self._cv)
            with open('meta.hist', 'a') as hst:
                for f in self._cv:
                    hst.write(f' {float(f)}')
                hst.write('\n')


class Qlvar:

    def __init__(self, i, env, cutoff=4., l=[4, 6]):
        """
        i:      index of atom for which ql will be calculated
        env:    type of atoms (Z) in the environment which contribute to ql
        cutoff: cutoff for atoms in the neighborhood of i
        l:      angular indices for ql
        """
        self.i = i
        self.env = env
        self.var = Ql(max(l), cutoff)
        self.l = l

    def __call__(self, numbers, xyz, cell, pbc, nl):
        nei_i, off = nl.get_neighbors(self.i)
        off = torch.from_numpy(off).type(xyz.type())
        off = (off[..., None]*cell).sum(dim=1)
        env = numbers[nei_i] == self.env
        r_ij = xyz[nei_i[env]] - xyz[self.i] + off[env]
        c = self.var(r_ij)
        return c.index_select(0, torch.tensor(self.l))
