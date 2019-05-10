
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module, Parameter
from torch.distributions import MultivariateNormal, LowRankMultivariateNormal
from theforce.regression.core import White
from theforce.regression.algebra import jitcholesky


class ConstMean(Module):

    def __init__(self, c=0., requires_grad=True):
        super().__init__()
        self.c = Parameter(torch.tensor(c), requires_grad=requires_grad)

    def forward(self, X):
        return torch.ones((X.size(0),)) * self.c

    def extra_repr(self):
        print('c: {}'.format(self.c))


class Covariance(Module):

    def __init__(self, kernels):
        super().__init__()
        self.kernels = (kernels if hasattr(kernels, '__iter__')
                        else (kernels,))

    def calculate(self, x=None, xx=None):
        return torch.stack([kern(x=x, xx=xx) for kern in self.kernels]).sum(dim=0)

    def diag(self, x=None):
        return torch.stack([kern.diag(x=x) for kern in self.kernels]).sum(dim=0)

    def forward(self, x=None, xx=None):
        return self.calculate(x=x, xx=xx)


class Inducing(Covariance):

    def __init__(self, kernels, x, num=None, learn=False, signal=5e-2):
        super().__init__(kernels)
        self.xind = Parameter(x.clone() if num is None else
                              x[torch.randint(0, x.size(0), (num,))],
                              requires_grad=learn)
        self.white = White(signal=signal)
        self.white._signal.requires_grad = True

    def extra_repr(self):
        print('num of inducing points: {}'.format(self.xind.size(0)))

    def decompose(self, x=None, xx=None):
        x_in = x is not None
        xx_in = xx is not None
        if not x_in and not xx_in:
            left = torch.ones(0, self.xind.size(0))
            right = left.t()
        elif x_in and not xx_in:
            left = self.calculate(x, self.xind)
            right = left.t()
        elif xx_in and not x_in:
            right = self.calculate(self.xind, xx)
            left = right.t()
        elif x_in and xx_in:
            left = self.calculate(x, self.xind)
            if x.shape == xx.shape and torch.allclose(x, xx):
                right = left.t()
            else:
                right = self.calculate(self.xind, xx)
        chol, _ = jitcholesky(self.calculate(self.xind, self.xind))
        return left, chol.inverse(), right

    def cov_factor(self, x):
        L, M, R = self.decompose(x=x)
        Q = L @ M.t()
        cov_loss = 0.5*(self.diag(x).sum() - torch.einsum('ij,ij', Q, Q))             / self.white.diag()
        return Q, self.white.diag(x), cov_loss

    def forward(self, x=None, xx=None):
        L, M, R = self.decompose(x=x, xx=xx)
        return L @ M.t() @ M @ R + self.white(x=x, xx=xx)


class GaussianProcess(Module):

    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov

    def forward(self, x):
        if hasattr(self.cov, 'cov_factor'):
            Q, diag, self.covariance_loss = self.cov.cov_factor(x)
            return LowRankMultivariateNormal(self.mean(x), Q, diag)
        else:
            self.covariance_loss = 0
            return MultivariateNormal(self.mean(x), covariance_matrix=self.cov(x))

    def loss(self, x, y):
        return -self(x).log_prob(y) + self.covariance_loss


class PosteriorGP(Module):

    def __init__(self, gp, X, Y):
        super().__init__()
        self.xind = X
        self.gp = gp
        p = gp(X)
        self.mu = p.precision_matrix @ (Y-p.loc)

    def mean(self, X):
        mean = self.gp.mean(X)
        cov = self.gp.cov(X, self.xind)
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
        #p = gp(X)
        #loss = -p.log_prob(Y)
        loss = gp.loss(X, Y)
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
    #cov = Covariance((SquaredExp(dim=dim), White(1e-2)))
    cov = Inducing(SquaredExp(dim=dim), X, 10, learn=False)
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

