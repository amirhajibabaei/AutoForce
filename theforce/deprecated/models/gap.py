import warnings

import ase
import numpy as np
import torch
from torch.distributions import LowRankMultivariateNormal
from torch.nn import Module, Parameter

from theforce.descriptor.clustersoap import ClusterSoap
from theforce.descriptor.radial_funcs import quadratic_cutoff
from theforce.descriptor.sesoap import SeSoap
from theforce.regression.algebra import (
    free_form,
    jitcholesky,
    low_rank_factor,
    positive,
    sum_packed_dim,
)
from theforce.regression.kernels import RBF
from theforce.util.tensors import SparseTensor, stretch_tensor

torch.set_default_tensor_type(torch.DoubleTensor)


def unnamed_operation(s):
    s._sort(key=1)
    s._split()
    size = max(s.i_max, s.j_max) + 1  # number of particles
    h = torch.zeros(s.shape[0], s.shape[0], size, size)
    for i, j, a in zip(*[s.i, s.j, s.a]):
        b = (stretch_tensor(a, (0, 1)) * stretch_tensor(a, (1, 2))).sum(dim=-1)
        k, l = torch.broadcast_tensors(
            i[
                None,
            ],
            i[:, None],
        )
        h[:, :, k, l] += b
    return h.permute(2, 3, 0, 1)


class GAP(Module):
    def __init__(self, lmax, nmax, cutoff):
        super(GAP, self).__init__()
        self.csoap = ClusterSoap(SeSoap(lmax, nmax, quadratic_cutoff(cutoff)))
        self.data = []

    def add_data(self, cluster):

        # configure
        if type(cluster) == tuple or type(cluster) == list:
            pbc, cell, positions, atomic_numbers, energy, forces = cluster
        elif isinstance(cluster, ase.Atoms):
            pbc, cell, positions = cluster.pbc, cluster.cell, cluster.positions
            atomic_numbers = cluster.get_atomic_numbers()
            energy = cluster.get_potential_energy()
            forces = cluster.get_forces()

        # atomic numbers
        if atomic_numbers is not None:
            if np.any(atomic_numbers != atomic_numbers[0]):
                raise NotImplementedError(
                    "heterogeneous atomic numbers are not implemented yet"
                )
            else:
                atomic_numbers = None

        # descriptors
        p, _s = self.csoap.describe(pbc, cell, positions, iderive=True, jderive=True)
        p = torch.as_tensor(p)
        s = SparseTensor(shape=(self.csoap.soap.dim, 0, 3))
        s.add(_s.i, _s.j, _s.a)
        h = unnamed_operation(s)
        s._cat()

        # apply torch.as_tensor
        if energy is not None:
            energy = torch.as_tensor([energy])
        if forces is not None:
            forces = torch.as_tensor(forces)

        # add to data
        self.data += [(p, s, h, energy, forces)]

        self.new_data = 1

    def build_X(self):
        self.X = torch.cat([a[0] for a in self.data])
        self.new_data = 0

    def select_Z(self, num_inducing):
        if not hasattr(self, "X") or self.new_data:
            self.build_X()
        rnd = torch.randint(len(self.data), (num_inducing,))
        Z = self.X[rnd]
        return Z

    def extend_Z(self, num):
        self.Z = Parameter(torch.cat([self.Z, self.select_Z(num)]), requires_grad=False)

    def parameterize(self, num_inducing, use_energies=1, use_forces=1, kern=RBF):
        if not hasattr(self, "parameterized") or not self.parameterized:
            # kernel param
            self._noise = Parameter(torch.tensor(1.0))
            self.kern = kern(torch.ones(self.csoap.soap.dim), torch.tensor(1.0))

            # inducing
            self.Z = Parameter(self.select_Z(num_inducing), requires_grad=False)

            # flags
            self.parameterized = 1
            self.use_energies = use_energies
            self.use_forces = use_forces
        else:
            warnings.warn("a second attempt to parameterize model is ignored!")

    def covariances(self):

        for (p, s, h, energy, forces) in self.data:

            # TODO: d_dx, d_dxdxx are only needed if forces are present
            zx, _, d_dx, _ = self.kern.matrices(self.Z, p, False, True, False)
            xx, _, _, d_dxdxx = self.kern.matrices(p, p, False, False, True)

            if self.use_energies and p is not None and energy is not None:
                ZX = zx.sum(dim=-1)
                diag = xx.sum()
                yield ZX.view(-1, 1), diag.view(1), energy.view(1)

            if (
                self.use_forces
                and s is not None
                and forces is not None
                and h is not None
            ):
                temp = -(d_dx[:, s.i, :, None] * s.a.permute(1, 0, 2)).sum(dim=2)
                m = self.Z.size(0)
                ZF = torch.zeros(m, *forces.size()).index_add(1, s.j, temp).view(m, -1)
                sum_diag = (d_dxdxx * h).sum()
                yield ZF, sum_diag.view(1), forces.view(-1)

    def matrices(self):
        ZZ, _, _, _ = self.kern.matrices(self.Z, self.Z)
        ZX, diag, Y = zip(*self.covariances())
        return ZZ, torch.cat(ZX, dim=1), torch.cat(diag), torch.cat(Y)

    def forward(self):

        # covariances
        ZZ, ZX, diag, Y = self.matrices()
        tr = diag.sum()
        noise = positive(self._noise)

        # trace term
        Q, _, ridge = low_rank_factor(ZZ, ZX)
        trace = 0.5 * (tr - torch.einsum("ij,ij", Q, Q)) / noise**2

        # low rank MVN
        p = LowRankMultivariateNormal(
            torch.zeros_like(Y), Q.t(), torch.ones_like(Y) * noise**2
        )

        # loss
        loss = -p.log_prob(Y) + trace
        return loss

    def train(self, steps=100, optimizer=None, lr=0.1, greedy=None):

        if not self.parameterized:
            warnings.warn("model is not parameterized yet! returned without training!")
            return

        if not hasattr(self, "losses"):
            self.losses = []
            self.starts = []
        self.starts += [len(self.losses)]

        if optimizer is None:
            if not hasattr(self, "optimizer"):
                self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            optimizer = self.optimizer

        for _ in range(steps):
            optimizer.zero_grad()
            loss = self.forward()
            self.losses += [loss.data]
            loss.backward()
            optimizer.step()
            if greedy is not None:
                self.greedy(steps=greedy)
        optimizer.zero_grad()  # NOTE: maybe unnecessary

        print("trained for {} steps".format(steps))

        self.ready = 0

    def greedy(self, steps=10):
        if not hasattr(self, "X") or self.new_data:
            self.build_X()
        n = self.X.size(0)
        m = self.Z.size(0)
        with torch.no_grad():
            for _ in range(steps):
                randX = torch.randint(n, (1,))
                randZ = torch.randint(m, (1,))
                Z = self.Z[randZ]
                self.Z[randZ] = self.X[randX]
                loss = self.forward()
                if loss < self.losses[-1]:
                    self.losses += [loss.data]
                else:
                    self.Z[randZ] = Z

        self.ready = 0

    def evaluate(self):

        ZZ, ZX, _, Y = self.matrices()
        XZ = ZX.t()
        noise = positive(self._noise)

        # numerically stable calculation of _mu
        L, ridge = jitcholesky(ZZ, jitbase=2)
        A = torch.cat((XZ, noise * L.t()))
        Y = torch.cat((Y, torch.zeros(self.Z.size(0), dtype=Y.dtype)))
        Q, R = torch.qr(A)
        self._mu = torch.mv(R.inverse(), torch.mv(Q.t(), Y))

        # inducing function values (Z, u)
        self.u = torch.mv(ZZ, self._mu)

        # TODO: predicted covariance

        self.ready = 1

    def predict(self, cluster):
        if not hasattr(self, "ready") or not self.ready:
            self.evaluate()

        # configure
        if type(cluster) == tuple or type(cluster) == list:
            pbc, cell, positions, atomic_numbers = cluster
        elif isinstance(cluster, ase.Atoms):
            pbc, cell, positions = cluster.pbc, cluster.cell, cluster.positions
            atomic_numbers = cluster.get_atomic_numbers()

        # descriptors
        p, _s = self.csoap.describe(pbc, cell, positions)
        p = torch.as_tensor(p)
        s = SparseTensor(shape=_s.shape)
        s.add(_s.i, _s.j, _s.a)
        s._cat()

        # covariances
        ZX, _, d_dx, _ = self.kern.matrices(self.Z, p, False, True, False)
        temp = -(d_dx[:, s.i, :, None] * s.a.permute(1, 0, 2)).sum(dim=2)
        m = self.Z.size(0)
        ZF = torch.zeros(m, *positions.shape).index_add(1, s.j, temp).view(m, -1)
        XZ = torch.cat([ZX, ZF], dim=1).t()

        # predict
        mu = torch.mv(XZ, self._mu)
        energy = mu[0 : p.size(0)].sum()
        forces = mu[p.size(0) :].view(-1, 3)
        return energy, forces
