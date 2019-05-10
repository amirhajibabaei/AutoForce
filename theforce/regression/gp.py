
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module, Parameter
from torch.distributions import MultivariateNormal


class ConstMean(Module):

    def __init__(self, c=0., requires_grad=True):
        super().__init__()
        self.c = Parameter(torch.tensor(c), requires_grad=requires_grad)

    def forward(self, X):
        return torch.ones((X.size(0),)) * self.c


class Covariance(Module):

    def __init__(self, kernels):
        super().__init__()
        self.kernels = (kernels if hasattr(kernels, '__iter__')
                        else (kernels,))

    def forward(self, x=None, xx=None):
        return torch.stack([kern(x=x, xx=xx) for kern in self.kernels]).sum(dim=0)


class GaussianProcess(Module):

    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov

    def forward(self, x):
        return MultivariateNormal(self.mean(x), covariance_matrix=self.cov(x))


class PosteriorGP(Module):

    def __init__(self, gp, X, Y):
        super().__init__()
        self.X = X
        self.gp = gp
        p = gp(X)
        self.mu = p.precision_matrix @ (Y-p.loc)

    def mean(self, X):
        mean = self.gp.mean(X)
        cov = self.gp.cov(X, self.X)
        return mean + cov @ self.mu

    def cov(self, X):
        raise NotImplementedError('Covariance has not been implemented yet!')

    def forward(self, X):
        raise NotImplementedError(''.join(('Similar to GaussianProcess class, this should return',
                                           'a MultivariateNormal instance which is not implemented yet')))


def train_gp(gp, X, Y, steps=100):
    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad()
        p = gp(X)
        loss = -p.log_prob(Y)
        loss.backward()
        optimizer.step()


def test():
    from theforce.regression.core import SquaredExp, White
    import pylab as plt
    get_ipython().run_line_magic('matplotlib', 'inline')

    # data
    n = 100
    dim = 1
    torch.random.manual_seed(534654647)
    X = (torch.rand(n, dim)-0.5)*10
    Y = (-(X**2).sum(dim=-1)).exp()

    # model
    cov = Covariance((SquaredExp(dim=dim), White(1e-2)))
    gp = GaussianProcess(ConstMean(), cov)
    train_gp(gp, X, Y)
    gpr = PosteriorGP(gp, X, Y)
    with torch.no_grad():
        XX = torch.arange(-5.0, 5.0, 0.1).view(-1, 1)
        f = gpr.mean(XX)
    plt.scatter(X, Y)
    plt.scatter(XX, f)


if __name__ == '__main__':
    test()

