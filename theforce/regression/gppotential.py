
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module
from torch.distributions import MultivariateNormal, LowRankMultivariateNormal
from theforce.regression.core import LazyWhite
from theforce.regression.algebra import jitcholesky
from theforce.util.util import iterable
import copy


class EnergyForceKernel(Module):

    def __init__(self, similaritykernels):
        super().__init__()
        self.kernels = iterable(similaritykernels)
        self.params = [par for kern in self.kernels for par in kern.params]

    def forward(self, first, second=None, cov='energy_energy', inducing=None):
        sec = first if second is None else second
        if inducing is None:
            return getattr(self, cov)(first, sec)
        else:
            middle = getattr(self, 'energy_energy')(inducing, inducing)
            chol, _ = jitcholesky(middle)
            invchol = chol.inverse()
            lcov, rcov = cov.split('_')
            left = getattr(self, lcov+'_energy')(first, inducing) @ invchol.t()
            if second is None and rcov == lcov:
                right = left.t()
            else:
                right = invchol @ getattr(self, 'energy_'+rcov)(inducing, sec)
            return left, right

    def diag(self, data):
        return torch.cat([self.energy_energy(sys, sys).view(1) for sys in data] +
                         [self.forces_forces(sys, sys).diag() for sys in data])

    def energy_energy(self, first, second):
        return self.base_kerns(first, second, 'func')

    def forces_energy(self, first, second):
        return -self.base_kerns(first, second, 'leftgrad')

    def energy_forces(self, first, second):
        return -self.base_kerns(first, second, 'rightgrad')

    def forces_forces(self, first, second):
        return self.base_kerns(first, second, 'gradgrad')

    def base_kerns(self, first, second, operation):
        return torch.stack([kern(first, second, operation)
                            for kern in self.kernels]).sum(dim=0)


class GaussianProcessPotential(Module):

    def __init__(self, kernels):
        super().__init__()
        self.kern = EnergyForceKernel(kernels)
        self.noise = LazyWhite(signal=0.01, requires_grad=True)
        self.params = self.kern.params + self.noise.params

    def forward(self, data, inducing=None):
        if inducing is None:
            L = torch.cat([self.kern(data, cov='energy_energy'),
                           self.kern(data, cov='forces_energy')], dim=0)
            R = torch.cat([self.kern(data, cov='energy_forces'),
                           self.kern(data, cov='forces_forces')], dim=0)
            return MultivariateNormal(torch.zeros(L.size(0)),
                                      covariance_matrix=torch.cat([L, R], dim=1) +
                                      torch.eye(L.size(0))*self.noise.diag())
        else:
            Q = torch.cat([self.kern(data, cov='energy_energy', inducing=inducing)[0],
                           self.kern(data, cov='forces_forces', inducing=inducing)[0]], dim=0)
            return LowRankMultivariateNormal(torch.zeros(Q.size(0)), Q,
                                             torch.ones(Q.size(0))*self.noise.diag())

    def Y(self, data):
        return torch.cat([torch.tensor([sys.energy for sys in data])] +
                         [sys.forces.view(-1) for sys in data])

    def loss(self, data, Y=None, inducing=None):
        p = self(data, inducing=inducing)
        if hasattr(p, 'cov_factor'):
            covariance_loss = 0.5*(self.kern.diag(data).sum() - torch.einsum(
                'ij,ij', p.cov_factor, p.cov_factor))/self.noise.diag()
        else:
            covariance_loss = 0
        return -p.log_prob(self.Y(data) if Y is None else Y) + covariance_loss


class PosteriorPotential(Module):

    def __init__(self, gp, data, inducing=None):
        super().__init__()
        self.gp = gp
        p = gp(data, inducing)
        if inducing is None:
            self.X = copy.deepcopy(data)
            self.mu = p.precision_matrix @ (gp.Y(data)-p.loc)
        else:
            K = torch.cat([gp.kern(data, inducing, cov='energy_energy'),
                           gp.kern(data, inducing, cov='forces_energy')], dim=0)
            M = gp.kern.energy_energy(inducing, inducing)
            L, _ = jitcholesky(M)
            A = torch.cat([K, gp.noise.diag().sqrt()*L.t()], dim=0)
            Y = torch.cat([gp.Y(data), torch.zeros(L.size(0))], dim=0)
            Q, R = torch.qr(A)
            self.mu = R.inverse() @ Q.t() @ Y
            self.X = copy.deepcopy(inducing)
            self.inducing = 1

    def forward(self, test):
        A = self.gp.kern(test, self.X, cov='energy_energy')
        B = self.gp.kern(test, self.X, cov='forces_energy')
        if not hasattr(self, 'inducing'):
            A = torch.cat([A, self.gp.kern(test, self.X, cov='energy_forces')],
                          dim=1)
            B = torch.cat([B, self.gp.kern(test, self.X, cov='forces_forces')],
                          dim=1)

        energy = A @ self.mu
        forces = B @ self.mu
        return energy, forces.view(-1, 3)


def train_gpp(gp, X, inducing=None, steps=10, lr=0.1, Y=None):
    if not hasattr(gp, 'optimizer'):
        gp.optimizer = torch.optim.Adam(gp.params, lr=lr)

    for _ in range(steps):
        gp.optimizer.zero_grad()
        loss = gp.loss(X, Y, inducing)
        loss.backward()
        gp.optimizer.step()
    print('trained for {} steps'.format(steps))

