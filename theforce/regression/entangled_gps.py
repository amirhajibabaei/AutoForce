
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module, Parameter
from torch.distributions import MultivariateNormal
from theforce.regression.algebra import free_form, positive


class ConstMean(Module):

    def __init__(self, requires_grad=True, tasks=1):
        super().__init__()
        self.c = Parameter(torch.zeros(tasks), requires_grad=requires_grad)

    def forward(self, X, i):
        return torch.ones((X.size(0),)) * self.c[i]


class WhiteNoise(Module):

    def __init__(self, sigma, requires_grad=False):
        super().__init__()
        self.sigma = Parameter(torch.as_tensor(sigma),
                               requires_grad=requires_grad)

    def forward(self, x, i, xx=None, ii=None):
        if xx is None:
            return torch.eye(x.size(0)) * self.sigma[i]**2
        else:
            return torch.zeros(x.size(0), xx.size(0))


class RBF(Module):

    def __init__(self, dim, tasks):
        super().__init__()
        # TODO: this is redundant as only diagonal elements are used
        self._scale = Parameter(free_form(torch.ones(tasks, tasks, dim)))
        self._variance = Parameter(free_form(torch.ones(tasks, tasks)))

    def forward(self, x, i, xx=None, ii=None):
        scale = positive(self._scale)
        variance = positive(self._variance)
        if xx is None:
            xx = x
            ii = i
        j, jj = torch.broadcast_tensors(i[:, None], ii[None])
        r = (x[:, None]-xx[None]) / scale[j, jj]
        cov = (-(r**2).sum(dim=-1)/2).exp() * variance[j, jj]
        cov[j != jj] = 0.0
        return cov


class GaussianProcess(Module):

    def __init__(self, mean, kernels):
        super().__init__()
        self.mean = mean
        self.kernels = kernels

    def covariance_matrix(self, X, i, XX=None, ii=None):
        cov = torch.stack([kern(X, i, xx=XX, ii=ii) for kern in
                           (self.kernels if hasattr(self.kernels, '__iter__')
                            else (self.kernels,))]).sum(dim=0)
        return cov

    def forward(self, X, i, t=None):
        loc = self.mean(X, i)
        cov = self.covariance_matrix(X, i)
        if t is not None:
            loc = t @ loc
            cov = t @ cov @ t.t()
        return MultivariateNormal(loc, covariance_matrix=cov)


def train_gp(gp, X, i, Y, t=None, steps=100):
    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad()
        p = gp(X, i, t=t)
        loss = -p.log_prob(Y)
        loss.backward()
        optimizer.step()


class GPRegression(Module):

    def __init__(self, gp, X, i, Y, t=None):
        super().__init__()
        self.X = X
        self.i = i
        self.gp = gp
        p = gp(X, i, t=t)
        self.mu = t.t() @ p.precision_matrix @ (Y-p.loc)

    def forward(self, X, i):
        mean = self.gp.mean(X, i)
        cov = self.gp.covariance_matrix(X, i, self.X, self.i)
        return mean + cov @ self.mu


def test():
    import pylab as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    # generate two independent sets of data
    dim = 1
    n, m, l = 29, 37, 19
    i1 = torch.zeros(n).long()
    X1 = (torch.rand(n, dim)-0.5) * 6
    Y1 = X1.sin()

    i2 = torch.ones(m).long()
    X2 = (torch.rand(m, dim)-0.5) * 6
    Y2 = X2.cos()

    # concatenate the to sets
    i = torch.cat([i1, i2])
    X = torch.cat([X1, X2])
    Y = torch.cat([Y1, Y2]).view(-1)

    # mix Y values by a random matrix multiplication
    mix = torch.rand(l, n+m)
    Y = (mix @ Y).view(-1)

    # regression
    kern = RBF(dim, 2)
    mean = ConstMean(tasks=2)
    wn = WhiteNoise([0.01, 0.01])
    gp = GaussianProcess(mean, [kern, wn])
    train_gp(gp, X, i, Y, t=mix)
    gpr = GPRegression(gp, X, i, Y, t=mix)
    with torch.no_grad():
        y1 = gpr(X1, i1)
        y2 = gpr(X2, i2)
    plt.scatter(X1.numpy(), y1.numpy())
    plt.scatter(X2.numpy(), y2.numpy())


if __name__ == '__main__':
    test()

