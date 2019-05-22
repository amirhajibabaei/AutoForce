
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module
from torch.distributions import MultivariateNormal


class GaussianProcessPotential(Module):

    def __init__(self, kernel):
        super().__init__()
        self.kern = kernel
        self.params = kernel.params

    def forward(self, data):
        ee = self.kern(data, data, 'func')
        ef = -self.kern(data, data, 'rightgrad')
        fe = -self.kern(data, data, 'leftgrad')
        ff = self.kern(data, data, 'gradgrad')
        cov = torch.cat([torch.cat([ee, ef], dim=1),
                         torch.cat([fe, ff], dim=1)], dim=0)
        loc = torch.zeros(cov.size(0))
        return MultivariateNormal(loc, covariance_matrix=cov)

    def Y(self, data):
        return torch.cat([torch.tensor([sys.energy for sys in data])] +
                         [sys.forces.view(-1) for sys in data])

    def loss(self, data, Y=None):
        return -self(data).log_prob(self.Y(data) if Y is None else Y)


class PosteriorPotential(Module):

    def __init__(self, gp, data):
        super().__init__()
        self.data = data
        self.gp = gp
        p = gp(data)
        self.mu = p.precision_matrix @ (gp.Y(data)-p.loc)

    def forward(self, test):
        ee = self.gp.kern(test, self.data, 'func')
        ef = -self.gp.kern(test, self.data, 'rightgrad')
        energy = torch.cat([ee, ef], dim=1) @ self.mu
        fe = -self.gp.kern(test, self.data, 'leftgrad')
        ff = self.gp.kern(test, self.data, 'gradgrad')
        forces = torch.cat([fe, ff], dim=1) @ self.mu
        return energy, forces.view(-1, 3)


def train_gpp(gp, X, Y=None, steps=10, lr=0.1):
    if not hasattr(gp, 'optimizer'):
        gp.optimizer = torch.optim.Adam(gp.params, lr=lr)

    for _ in range(steps):
        gp.optimizer.zero_grad()
        loss = gp.loss(X, Y)
        loss.backward()
        gp.optimizer.step()
    print('trained for {} steps'.format(steps))

