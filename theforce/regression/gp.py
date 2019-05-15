
# coding: utf-8

# In[ ]:


""" experimental """
import torch
from torch.nn import Module, Parameter
from torch.distributions import MultivariateNormal, LowRankMultivariateNormal
from theforce.regression.core import LazyWhite
from theforce.regression.algebra import jitcholesky
"""
NOTES:
1. Hessian has to be multiplied by -1 because kernels return 
derivatives wrt r and not x, x'.
2. When Y are grad data, (-1) is multiplied to mu because X
will play the role of xx in PosteriorGP.mean. This will also
satisfy the -1 needed in PosteriorGP.grad.

TODO:
1. Make Inducing kernel work with grad data.
"""


class ConstMean(Module):

    def __init__(self, c=0., requires_grad=True):
        super().__init__()
        self.c = Parameter(torch.tensor(c), requires_grad=requires_grad)

    def forward(self, X, operation='func'):
        if operation == 'func':
            return torch.ones((X.size(0),)) * self.c
        elif operation == 'grad':
            return torch.zeros_like(X)
        else:
            raise NotImplementedError(operation+'is not implemented!')

    def extra_repr(self):
        print('c: {}'.format(self.c))


class Covariance(Module):

    def __init__(self, kernels):
        super().__init__()
        self.kernels = (kernels if hasattr(kernels, '__iter__')
                        else (kernels,))
        self.params = [par for kern in self.kernels for par in kern.params]

    def calculate(self, x=None, xx=None, operation='func'):
        return torch.stack([kern(x=x, xx=xx, operation=operation)
                            for kern in self.kernels]).sum(dim=0)

    def diag(self, x=None):
        return torch.stack([kern.diag(x=x) for kern in self.kernels]).sum(dim=0)

    def forward(self, x=None, xx=None, operation='func'):
        return self.calculate(x=x, xx=xx, operation=operation)


class Inducing(Covariance):        # TODO 1.

    def __init__(self, kernels, x, num=None, learn=False, signal=5e-2):
        super().__init__(kernels)
        self.xind = Parameter(x.clone() if num is None else
                              x[torch.randint(0, x.size(0), (num,))],
                              requires_grad=learn)
        self.white = LazyWhite(dim=x.size(0), signal=signal)
        self.white._signal.requires_grad = True
        self.params += [self.white._signal, self.xind]

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

    def forward(self, x=None, xx=None, operation='placeholder=func'):
        L, M, R = self.decompose(x=x, xx=xx)
        return L @ M.t() @ M @ R + self.white(x=x, xx=xx)


class GaussianProcess(Module):

    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.params = self.cov.params + list(self.mean.parameters())

    def forward(self, x, op='func'):
        self.covariance_loss = 0
        if op == 'func':
            if hasattr(self.cov, 'cov_factor'):
                Q, diag, self.covariance_loss = self.cov.cov_factor(x)
                return LowRankMultivariateNormal(self.mean(x), Q, diag)
            else:
                return MultivariateNormal(self.mean(x),
                                          covariance_matrix=self.cov(x))
        elif op == 'grad':
            if hasattr(self.cov, 'cov_factor'):
                raise NotImplementedError(
                    'Inducing kernel is not implemented for grads yet!')
            else:
                cov = (-1)*self.cov(x, operation='gradgrad').reshape(     # NOTE 1.
                    x.numel(), x.numel())
                return MultivariateNormal(self.mean(x, operation='grad').reshape(-1),
                                          covariance_matrix=cov)

    def loss(self, x, y):
        if y.dim() == 1:
            return -self(x, op='func').log_prob(y) + self.covariance_loss
        elif y.shape == x.shape:
            return -self(x, op='grad').log_prob(y.reshape(-1)) + self.covariance_loss
        else:
            raise RuntimeError('Shape of Y is not consistent!')


class PosteriorGP(Module):

    def __init__(self, gp, X, Y):
        super().__init__()
        self.x = X
        self.gp = gp
        if X.shape == Y.shape:
            self.Y_type = 'grad'
            sign = -1                     # NOTE 2.
        else:
            self.Y_type = 'func'
            sign = 1
        p = gp(X, op=self.Y_type)
        self.mu = p.precision_matrix @ (Y.reshape(-1)-p.loc) * sign

    def mean(self, X):
        mean = self.gp.mean(X)
        cov = self.gp.cov(X, self.x, operation=self.Y_type)
        return mean + cov.reshape(-1, self.mu.size(0)) @ self.mu

    def grad(self, X):
        gradmean = self.gp.mean(X, operation='grad').reshape(-1)
        if self.Y_type == 'func':
            cov = self.gp.cov(X, self.x, operation='grad').permute(0, 2, 1)
        elif self.Y_type == 'grad':
            cov = self.gp.cov(X, self.x, operation='gradgrad')
        return (gradmean + cov.reshape(-1, self.mu.size(0)) @ self.mu).reshape_as(X)

    def cov(self, X):
        raise NotImplementedError('Covariance has not been implemented yet!')

    def forward(self, X):
        raise NotImplementedError(''.join(('Similar to GaussianProcess class, this should return',
                                           'a MultivariateNormal instance which is not implemented yet')))


def train_gp(gp, X, Y, steps=100, lr=0.1):
    optimizer = torch.optim.Adam(gp.params, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = gp.loss(X, Y)
        loss.backward()
        optimizer.step()


def test_basic():
    from theforce.regression.core import SquaredExp, LazyWhite
    import pylab as plt
    get_ipython().run_line_magic('matplotlib', 'inline')

    # data
    n = 100
    dim = 1
    torch.random.manual_seed(534654647)
    X = (torch.rand(n, dim)-0.5)*10
    #Y = (-(X**2).sum(dim=-1)).exp()
    Y = X.sin().sum(dim=1)

    # model
    #cov = Covariance((SquaredExp(dim=dim), LazyWhite(dim=dim, signal=1e-2)))
    cov = Inducing(SquaredExp(dim=dim), X, 9, learn=True)
    gp = GaussianProcess(ConstMean(), cov)
    train_gp(gp, X, Y, steps=500, lr=0.1)
    gpr = PosteriorGP(gp, X, Y)
    with torch.no_grad():
        XX = torch.linspace(X.min(), X.max(), 100).view(-1, 1)
        f = gpr.mean(XX)
    plt.scatter(X, Y)
    plt.scatter(XX, f)
    if hasattr(cov, 'xind'):
        plt.scatter(cov.xind.detach().numpy(),
                    gpr.mean(cov.xind).detach().numpy(),
                    marker='x', s=200)


def test_grad():
    from theforce.regression.core import SquaredExp, LazyWhite
    n = 77
    dim = 1
    torch.random.manual_seed(534654647)
    X = (torch.rand(n, dim)-0.5)*10
    Y = X.sin().sum(dim=1)
    dY = X.cos()
    Y_data = dY             # use Y or dY

    kernels = (SquaredExp(dim=dim), LazyWhite(dim=dim, signal=0.01))

    # model
    cov = Covariance(kernels)
    gp = GaussianProcess(ConstMean(), cov)
    train_gp(gp, X, Y_data, steps=100)
    gpr = PosteriorGP(gp, X, Y_data)

    if 1:
        import pylab as plt
        get_ipython().run_line_magic('matplotlib', 'inline')
        with torch.no_grad():
            XX = torch.linspace(X.min(), X.max(), 50).view(-1, 1)
            f = gpr.mean(XX)
            df = gpr.grad(XX)

        plt.scatter(X, Y, label='Y')
        plt.scatter(X, dY, label='dY')
        plt.scatter(XX, f, label='f')
        plt.scatter(XX, df, label='df')
        if hasattr(cov, 'xind'):
            plt.scatter(cov.xind.detach().numpy(),
                        gpr.mean(cov.xind).detach().numpy(),
                        marker='x', s=200)
        plt.legend()


def test_multidim():
    import torch
    from theforce.regression.core import SquaredExp, LazyWhite
    from theforce.regression.gp import ConstMean, Covariance, GaussianProcess, PosteriorGP, train_gp

    def dummy_data(X):
        Y = (-X**2).sum(dim=1).exp()
        dY = -2*X*Y[:, None]
        return Y, dY

    X = (torch.rand(20, 2)-0.5)*2
    Y, dY = dummy_data(X)
    for data_Y in [Y, dY]:
        cov = Covariance((SquaredExp(dim=2), LazyWhite(dim=2, signal=0.01)))
        gp = GaussianProcess(ConstMean(), cov)
        train_gp(gp, X, data_Y, steps=500)
        gpr = PosteriorGP(gp, X, data_Y)
        assert (gpr.mean(X)-Y).var().sqrt() < 0.05
        assert (gpr.grad(X)-dY).var().sqrt() < 0.05


if __name__ == '__main__':
    #test_basic()
    #test_grad()
    test_multidim()

