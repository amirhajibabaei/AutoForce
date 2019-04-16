
# coding: utf-8

# In[ ]:


from theforce.descriptor.sesoap import SeSoap
from theforce.descriptor.radial_funcs import quadratic_cutoff
from theforce.descriptor.clustersoap import ClusterSoap
from theforce.regression.kernels import RBF
from theforce.regression.algebra import low_rank_factor, jitcholesky
from theforce.regression.algebra import positive, free_form, sum_packed_dim
import ase
import numpy as np
import torch
from torch.nn import Module, Parameter
from torch.distributions import LowRankMultivariateNormal
import warnings
torch.set_default_tensor_type(torch.DoubleTensor)


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
                    'heterogeneous atomic numbers are not implemented yet')
            else:
                atomic_numbers = None

        # descriptors
        if forces is not None:
            p, q, indices = self.csoap.descriptors_derivatives(pbc, cell, positions,
                                                               sumj=False, jsorted=True)
        elif energy is not None:
            p = self.csoap.descriptors(pbc, cell, positions)
            q, indices = None, None
        else:
            p, q, indices = None, None, None

        # apply torch.as_tensor
        if energy is not None:
            energy = torch.as_tensor([energy])
        if forces is not None:
            forces = torch.as_tensor(forces)
        if p is not None:
            p = torch.as_tensor(p)
        if q is not None:
            q = [torch.as_tensor(v) for v in q]
            indices = [torch.as_tensor(v) for v in indices]

        # add to data
        self.data += [(p, q, indices, energy, forces)]

    def parameterize(self, num_inducing, use_energies=1, use_forces=1):
        # kernel param
        self._noise = Parameter(torch.tensor(1.))
        self.kern = RBF(torch.ones(self.csoap.soap.dim), torch.tensor(1.))

        # inducing
        rnd = torch.randint(len(self.data), (num_inducing,))
        Z = torch.cat([self.data[k][0][torch.randint(self.data[k][0].size()[0], (1,))]
                       for k in rnd], dim=0)
        self.Z = Parameter(Z, requires_grad=False)

        # flags
        self.parameterized = 1
        self.use_energies = use_energies
        self.use_forces = use_forces

    def covariances(self):

        for (p, q, indices, energy, forces) in self.data:

            # TODO: d_dx, d_dxdxx are only needed if forces are present
            # TODO: d_dxdxx is not fully needed at a given instance
            zx, _, d_dx, _ = self.kern.matrices(self.Z, p, False,
                                                True, False)
            xx, _, _, d_dxdxx = self.kern.matrices(p, p, False,
                                                   False, True)

            if self.use_energies and p is not None and energy is not None:
                ZX = zx.sum(dim=-1)
                diag = xx.sum()
                yield ZX.view(-1, 1), diag.view(1), energy.view(1)

            if self.use_forces and q is not None and forces is not None and indices is not None:
                for i in range(len(q)):
                    dxj_dri, j = q[i], indices[i]
                    ZF = -torch.einsum('ijp,pjm->im', d_dx[:, j], dxj_dri)
                    diag = torch.einsum('qim,ijqp,pjm->m',
                                        dxj_dri, d_dxdxx[j][:, j], dxj_dri)
                    yield ZF, diag.view(3), forces[i].view(3)

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
        trace = 0.5*(tr - torch.einsum('ij,ij', Q, Q))/noise**2

        # low rank MVN
        p = LowRankMultivariateNormal(torch.zeros_like(Y), Q.t(),
                                      torch.ones_like(Y)*noise**2)

        # loss
        loss = -p.log_prob(Y) + trace
        return loss

    def train(self, steps=100, optimizer=None, lr=0.1):

        if not self.parameterized:
            warnings.warn(
                'model is not parameterized yet! returned without training!')
            return

        if not hasattr(self, 'losses'):
            self.losses = []
            self.starts = []
        self.starts += [len(self.losses)]

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            loss = self.forward()
            self.losses += [loss.data]
            loss.backward()
            optimizer.step()
        print('trained for {} steps'.format(steps))

        self.ready = 0

    def evaluate(self):

        ZZ, ZX, _, Y = self.matrices()
        XZ = ZX.t()
        noise = positive(self._noise)

        # numerically stable calculation of _mu
        L, ridge = jitcholesky(ZZ, jitbase=2)
        A = torch.cat((XZ, noise * L.t()))
        Y = torch.cat((Y, torch.zeros(self.Z.size()[0],
                                      dtype=Y.dtype)))
        Q, R = torch.qr(A)
        self._mu = torch.mv(R.inverse(), torch.mv(Q.t(), Y))

        # inducing function values (Z, u)
        self.u = torch.mv(ZZ, self._mu)

        # TODO: predicted covariance

        self.ready = 1

    def predict(self, cluster):
        if not hasattr(self, 'ready') or not self.ready:
            self.evaluate()

        # configure
        if type(cluster) == tuple or type(cluster) == list:
            pbc, cell, positions, atomic_numbers = cluster
        elif isinstance(cluster, ase.Atoms):
            pbc, cell, positions = cluster.pbc, cluster.cell, cluster.positions
            atomic_numbers = cluster.get_atomic_numbers()

        # descriptors
        p, q, indices = self.csoap.descriptors_derivatives(pbc, cell, positions,
                                                           sumj=False, jsorted=True)
        p = torch.as_tensor(p)
        q = [torch.as_tensor(v) for v in q]
        indices = [torch.as_tensor(v) for v in indices]

        # covariances
        ZX, _, d_dx, _ = self.kern.matrices(self.Z, p, False, True, False)
        ZF = []
        for i in range(len(q)):
            dxj_dri, j = q[i], indices[i]
            ZF += [-torch.einsum('ijp,pjm->im', d_dx[:, j], dxj_dri)]
        XZ = torch.cat([ZX, *ZF], dim=1).t()

        # predict
        mu = torch.mv(XZ, self._mu)
        energy = mu[0:p.size()[0]].sum()
        forces = mu[p.size()[0]:].view(-1, 3)
        return energy, forces

