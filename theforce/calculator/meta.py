# +
from theforce.util.kde import Gaussian_kde
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
