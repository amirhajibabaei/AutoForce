
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module
from torch.distributions import MultivariateNormal, LowRankMultivariateNormal
from theforce.regression.core import LazyWhite
from theforce.regression.algebra import jitcholesky
from theforce.util.util import iterable
from theforce.optimize.optimizers import ClampedSGD
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

    # diagonal elements:
    def diag(self, data, operation='energy'):
        return getattr(self, operation+'_diag')(data)

    def full_diag(self, data):
        return self.energy_forces_diag(data)

    def energy_forces_diag(self, data):
        return torch.cat([self.energy_diag(data), self.forces_diag(data)])

    def energy_diag(self, data):
        return self.base_kerns_diag(data, 'func')

    def forces_diag(self, data):
        return self.base_kerns_diag(data, 'gradgrad')

    def base_kerns_diag(self, data, operation):
        return torch.stack([kern.diag(data, operation)
                            for kern in self.kernels]).sum(dim=0)

    @property
    def state_args(self):
        return '[{}]'.format(', '.join([kern.state for kern in self.kernels]))

    @property
    def state(self):
        return 'EnergyForceKernel({})'.format(self.state_args)


class GaussianProcessPotential(Module):

    def __init__(self, kernels, noise=LazyWhite(signal=0.01, requires_grad=True)):
        super().__init__()
        self.kern = EnergyForceKernel(kernels)
        self.noise = noise
        self.params = self.kern.params + self.noise.params
        for i, kern in enumerate(self.kern.kernels):
            kern.name = 'kern_{}'.format(i)

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
        return torch.cat([torch.tensor([sys.target_energy for sys in data])] +
                         [sys.target_forces.view(-1) for sys in data])

    def loss(self, data, Y=None, inducing=None, logprob_loss=True, cov_loss=False):
        p = self(data, inducing=inducing)
        if hasattr(p, 'cov_factor'):
            if cov_loss:
                covariance_loss = 0.5*(self.kern.diag(data, 'full').sum() - torch.einsum(
                                       'ij,ij', p.cov_factor, p.cov_factor))/self.noise.diag()
            else:
                covariance_loss = 0
        else:
            covariance_loss = 0
        if logprob_loss:
            lp_loss = -p.log_prob(self.Y(data) if Y is None else Y)
        else:
            lp_loss = 0
        return lp_loss + covariance_loss

    @property
    def state_args(self):
        return ', '.join([self.kern.state_args, self.noise.state])

    @property
    def state(self):
        return 'GaussianProcessPotential({})'.format(self.state_args)


class PosteriorPotential(Module):

    def __init__(self, gp, data, inducing=None):
        super().__init__()
        with torch.no_grad():
            self.gp = gp
            p = gp(data, inducing)
            if inducing is None:
                self.X = copy.deepcopy(data)
                self.mu = p.precision_matrix @ (gp.Y(data)-p.loc)
                self.sig = p.precision_matrix
                self.has_target_forces = True
            else:
                K = torch.cat([gp.kern(data, inducing, cov='energy_energy'),
                               gp.kern(data, inducing, cov='forces_energy')], dim=0)
                M = gp.kern(inducing, inducing, cov='energy_energy')
                L, _ = jitcholesky(M)
                A = torch.cat([K, gp.noise.diag().sqrt()*L.t()], dim=0)
                Y = torch.cat([gp.Y(data), torch.zeros(L.size(0))], dim=0)
                Q, R = torch.qr(A)
                self.mu = R.inverse() @ Q.t() @ Y
                i = L.inverse()
                W = K @ i.t() @ i
                self.sig = W.t() @ p.precision_matrix @ W

                self.X = inducing
                inducing.set_per_atoms('target_energy', M @ self.mu)
                F = gp.kern(inducing, inducing, cov='forces_energy')
                F = (F @ self.mu).reshape(-1, 3)
                inducing.set_per_atom('target_forces', F)
                self.has_target_forces = False

    def forward(self, test, quant='energy', variance=False, enable_grad=False):
        shape = {'energy': (-1,), 'forces': (-1, 3)}
        with torch.set_grad_enabled(enable_grad):
            A = self.gp.kern(test, self.X, cov=quant+'_energy')
            if self.has_target_forces:
                A = torch.cat([A, self.gp.kern(test, self.X, cov=quant+'_forces')],
                              dim=1)
            out = (A @ self.mu).view(*shape[quant])
            if variance:
                var = (self.gp.kern.diag(test, quant) -
                       (A @ self.sig @ A.t()).diag()).view(*shape[quant])
                return out, var
            else:
                return out

    def predict(self, test, variance=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            A = self.gp.kern(test, self.X, cov='energy_energy')
            B = self.gp.kern(test, self.X, cov='forces_energy')
            if self.has_target_forces:
                A = torch.cat([A, self.gp.kern(test, self.X, cov='energy_forces')],
                              dim=1)
                B = torch.cat([B, self.gp.kern(test, self.X, cov='forces_forces')],
                              dim=1)

            energy = A @ self.mu
            forces = B @ self.mu

            out = (energy, forces.view(-1, 3))

            if variance:
                energy_var = (self.gp.kern.diag(test, 'energy') -
                              (A @ self.sig @ A.t()).diag())
                forces_var = (self.gp.kern.diag(test, 'forces') -
                              (B @ self.sig @ B.t()).diag())
                out += (energy_var, forces_var.view(-1, 3))

            return out


def train_gpp(gp, X, inducing=None, steps=10, lr=0.1, Y=None, logprob_loss=True, cov_loss=False,
              move=0.1, shake=0):
    if not logprob_loss and not cov_loss:
        raise RuntimeError('both loss terms are ignored!')

    if not hasattr(gp, 'optimizer'):
        gp.optimizer = torch.optim.Adam([{'params': gp.params}], lr=lr)
        if inducing is not None and inducing.trainable and not hasattr(inducing, 'optimizer'):
            inducing.optimizer = ClampedSGD(
                [{'params': inducing.params}], lr=move)

    for _ in range(steps):
        if inducing is not None and inducing.trainable:
            if shake > 0:
                inducing.shake(update=False)
            inducing.optimizer.zero_grad()
            inducing.update_nl_if_requires_grad(descriptors=gp.kern.kernels)
        gp.optimizer.zero_grad()
        loss = gp.loss(X, Y, inducing, logprob_loss, cov_loss)
        loss.backward()
        gp.optimizer.step()
        if inducing is not None and inducing.trainable:
            inducing.optimizer.step()

    report = []
    if inducing is None:
        report += ['Full']
    else:
        report += ['Sparse']
        if cov_loss:
            report += ['Variational-ELBO']
            if not logprob_loss:
                report += ['~only trace of Gram matrix considered']
        else:
            report += ['Projected-Process']
    report = ' '.join(report)
    print('trained for {} steps ({})'.format(steps, report))

