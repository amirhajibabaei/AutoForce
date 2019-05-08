
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module, Parameter
from torch.distributions import MultivariateNormal
from theforce.regression.algebra import free_form, positive


class ConstMean(Module):

    def __init__(self, c=0., requires_grad=True):
        super().__init__()
        self.c = Parameter(torch.tensor(c), requires_grad=requires_grad)

    def forward(self, X):
        return torch.ones((X.size(0),)) * self.c


class WhiteNoise(Module):

    def __init__(self, sigma, requires_grad=False):
        super().__init__()
        self.sigma = Parameter(torch.as_tensor(sigma),
                               requires_grad=requires_grad)

    def forward(self, x, xx=None):
        if xx is None:
            return torch.eye(x.size(0)) * self.sigma**2
        else:
            return torch.zeros(x.size(0), xx.size(0))


class RBF(Module):

    def __init__(self, dim):
        super().__init__()
        self._scale = Parameter(free_form(torch.ones(dim)))
        self._variance = Parameter(free_form(torch.tensor(1.0)))

    def forward(self, x, xx=None):
        scale = positive(self._scale)
        variance = positive(self._variance)
        if xx is None:
            xx = x
        r = (x[:, None]-xx[None]) / scale
        cov = (-(r**2).sum(dim=-1)/2).exp() * variance
        return cov


class GaussianProcess(Module):

    def __init__(self, mean, kernels):
        super().__init__()
        self.mean = mean
        self.kernels = kernels

    def covariance_matrix(self, X, XX=None):
        cov = torch.stack([kern(X, xx=XX) for kern in
                           (self.kernels if hasattr(self.kernels, '__iter__')
                            else (self.kernels,))]).sum(dim=0)
        return cov

    def forward(self, X):
        loc = self.mean(X)
        cov = self.covariance_matrix(X)
        return MultivariateNormal(loc, covariance_matrix=cov)


def train_gp(gp, X, Y, steps=100):
    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad()
        p = gp(X)
        loss = -p.log_prob(Y)
        loss.backward()
        optimizer.step()


class GPRegression(Module):

    def __init__(self, gp, X, Y):
        super().__init__()
        self.X = X
        self.gp = gp
        self.mu = gp(X).precision_matrix @ (Y-gp.mean(X))

    def forward(self, X):
        mean = self.gp.mean(X)
        cov = self.gp.covariance_matrix(X, self.X)
        return mean + cov @ self.mu


def test():
    import pylab as plt
    get_ipython().run_line_magic('matplotlib', 'inline')

    # data
    n = 100
    dim = 1
    torch.random.manual_seed(534654647)
    X = (torch.rand(n, dim)-0.5)*10
    Y = (-(X**2).sum(dim=-1)).exp()

    # model
    gp = GaussianProcess(ConstMean(), [RBF(dim), WhiteNoise(1e-2)])
    train_gp(gp, X, Y)
    gpr = GPRegression(gp, X, Y)
    with torch.no_grad():
        XX = torch.arange(-5.0, 5.0, 0.1).view(-1, 1)
        f = gpr(XX)
    plt.scatter(X, Y)
    plt.scatter(XX, f)


if __name__ == '__main__':
    test()

