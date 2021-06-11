
from theforce.descriptor.atoms import TorchAtoms, AtomsData, LocalsData
from theforce.regression.gppotential import EnergyForceKernel
from theforce.similarity.soap import SoapKernel
from theforce.descriptor.cutoff import PolyCut
from theforce.regression.kernel import Positive, DotProd, Normed


class SparseSampler:

    def __init__(self, cutoff, numbers, atomic_unit=None, lmax=2, nmax=2, exponent=4):
        self.cutoff = cutoff
        self.kern = EnergyForceKernel([SoapKernel(Positive(1.0)*Normed(DotProd()**exponent),
                                                  atomic_number, numbers, lmax, nmax,
                                                  PolyCut(cutoff), atomic_unit=atomic_unit)
                                       for atomic_number in numbers])
        for i, k in enumerate(self.kern.kernels):
            k.name = 'ss_{}'.format(i)
        self.sparse = None

    def load(self, trajname):
        locs = LocalsData(traj=trajname)
        locs.stage(self.kern.kernels)
        if self.sparse is None:
            self.sparse = locs
        else:
            self.sparse += locs

    def sample_from_atoms(self, atoms, tol=0.01):
        if self.sparse is None:
            self.sparse = LocalsData(X=[atoms[0]])
        for loc in atoms:
            if (self.kern(loc, self.sparse) > 1.-tol).any():
                continue
            else:
                self.sparse += loc

    def sample_from_data(self, data, tol=0.01):
        if self.sparse is None:
            self.sparse = LocalsData(X=[data[0][0]])
        for atoms in data:
            for loc in atoms:
                if (self.kern(loc, self.sparse) > 1.-tol).any():
                    continue
                else:
                    self.sparse += loc

    def sample_from_file(self, traj, tol=0.01):
        data = AtomsData(traj=traj, cutoff=cutoff,
                         descriptor=self.kern.kernels)
        self.sample_from_data(data, tol=tol)

    def to_traj(self, trajname, mode='w'):
        self.sparse.to_traj(trajname, mode=mode)

