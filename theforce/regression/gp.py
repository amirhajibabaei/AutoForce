
import torch
from torch.nn import Module, Parameter
from torch.distributions import MultivariateNormal, LowRankMultivariateNormal
from theforce.regression.core import LazyWhite
from theforce.regression.algebra import jitcholesky


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
    """
    Calculates the covariance matrix.
    A layer between stationary (base) kernels which depend
    only on r (=x-xx) and the data (x, xx).
    """

    def __init__(self, kernels):
        super().__init__()
        self.kernels = (kernels if hasattr(kernels, '__iter__')
                        else (kernels,))
        self.params = [par for kern in self.kernels for par in kern.params]

    def base_kerns(self, x=None, xx=None, operation='func'):
        return torch.stack([kern(x=x, xx=xx, operation=operation)
                            for kern in self.kernels]).sum(dim=0)

    def diag(self, x=None, operation='func'):
        if operation == 'func':
            sign = 1
        elif operation == 'gradgrad':
            sign = -1
        else:
            raise NotImplementedError('This shoud not happen!')
        return sign * torch.stack([kern.diag(x=x, operation=operation)
                                   for kern in self.kernels]).sum(dim=0)

    def forward(self, x=None, xx=None, operation='func'):
        if hasattr(self, operation):
            return getattr(self, operation)(x=x, xx=xx)
        else:
            return self.base_kerns(x=x, xx=xx, operation=operation)

    def leftgrad(self, x=None, xx=None):
        t = self.base_kerns(x=x, xx=xx, operation='grad').permute(0, 2, 1)
        return t.contiguous().view(t.size(0)*t.size(1), t.size(2))

    def rightgrad(self, x=None, xx=None):
        t = -self.base_kerns(x=x, xx=xx, operation='grad')
        return t.view(t.size(0), t.size(1)*t.size(2))

    def gradgrad(self, x=None, xx=None):
        t = -self.base_kerns(x=x, xx=xx, operation='gradgrad')
        return t.view(t.size(0)*t.size(1), t.size(2)*t.size(3))

    @property
    def state_args(self):
        return '[{}]'.format(', '.join([kern.state for kern in self.kernels]))

    @property
    def state(self):
        return 'Covariance({})'.format(self.state_args)


class Inducing(Covariance):

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

    def decompose(self, x=None, xx=None, operation='func'):
        if x is None and xx is None:
            x = torch.ones(0, self.xind.size(0))
            xx = torch.ones(0, self.xind.size(0))
        elif x is None:
            x = xx
        elif xx is None:
            xx = x
        left = (self.leftgrad(x, self.xind) if operation == 'leftgrad' or
                operation == 'gradgrad' else self.base_kerns(x, self.xind))
        right = (self.rightgrad(self.xind, xx) if operation == 'rightgrad' or
                 operation == 'gradgrad' else self.base_kerns(self.xind, xx))
        chol, _ = jitcholesky(self.base_kerns(self.xind, self.xind))
        return left, chol.inverse(), right

    def cov_factor(self, x, operation='func'):
        L, M, R = self.decompose(x=x, operation=operation)
        Q = L @ M.t()
        cov_loss = 0.5*(self.diag(x, operation=operation).sum() - torch.einsum('ij,ij', Q, Q))             / self.white.diag()
        return Q, self.white.diag(x, operation=operation), cov_loss

    def forward(self, x=None, xx=None, operation='func'):
        L, M, R = self.decompose(x=x, xx=xx, operation=operation)
        # NOTE 1. is white kernel needed here? it is buggy to include!
        # If it must be included in future, operation could be rightgrad, etc
        # which is not considered in the white level, and should be fixed.
        return L @ M.t() @ M @ R  # + self.white(x=x, xx=xx, operation=operation)


class GaussianProcess(Module):

    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.params = self.cov.params + list(self.mean.parameters())

    def forward(self, x, op='func'):
        self.covariance_loss = 0
        operation = {'func': 'func', 'grad': 'gradgrad'}
        if hasattr(self.cov, 'cov_factor'):
            Q, diag, self.covariance_loss = self.cov.cov_factor(
                x, operation=operation[op])
            return LowRankMultivariateNormal(self.mean(x, operation=op).view(-1), Q, diag)
        else:
            return MultivariateNormal(self.mean(x, operation=op).view(-1),
                                      covariance_matrix=self.cov(x, operation=operation[op]))

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
        self.dtype = 'grad' if X.shape == Y.shape else 'func'
        p = gp(X, op=self.dtype)
        self.mu = p.precision_matrix @ (Y.reshape(-1)-p.loc)
        self.kw = {('func', 'func'): 'func', ('func', 'grad'): 'rightgrad',
                   ('grad', 'func'): 'leftgrad', ('grad', 'grad'): 'gradgrad'}

    def loc(self, X, op='func'):
        mean = self.gp.mean(X, operation=op)
        cov = self.gp.cov(X, self.x, operation=self.kw[(op, self.dtype)])
        return mean + ((cov @ self.mu).reshape_as(X) if op == 'grad' else cov @ self.mu)

    def mean(self, X):
        return self.loc(X, op='func')

    def grad(self, X):
        return self.loc(X, op='grad')

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
        df = gpr.grad(XX)
    plt.scatter(X, Y, label='Y')
    plt.scatter(XX, f, label='f')
    plt.scatter(XX, df, label='df')
    if hasattr(cov, 'xind'):
        plt.scatter(cov.xind.detach().numpy(),
                    gpr.mean(cov.xind).detach().numpy(),
                    marker='x', s=200)
    plt.legend()


def test_grad():
    from theforce.regression.core import SquaredExp, LazyWhite
    n = 77
    dim = 1
    torch.random.manual_seed(534654647)
    X = (torch.rand(n, dim)-0.5)*10
    Y = X.sin().sum(dim=1)
    dY = X.cos()
    Y_data = dY             # use Y or dY

    # model
    #cov = Covariance((SquaredExp(dim=dim), LazyWhite(dim=dim, signal=0.01)))
    cov = Inducing((SquaredExp(dim=dim),), X, 8, learn=True)
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

    def dummy_data(X):
        Y = (-X**2).sum(dim=1).exp()
        dY = -2*X*Y[:, None]
        return Y, dY

    torch.random.manual_seed(56453435468)
    X = (torch.rand(100, 2)-0.5)*2
    Y, dY = dummy_data(X)
    for data_Y in [Y, dY]:
        cov1 = Covariance((SquaredExp(dim=2), LazyWhite(dim=2, signal=0.01)))
        cov2 = Inducing((SquaredExp(dim=2),), X, 19, learn=True)
        for cov in (cov1, cov2):
            gp = GaussianProcess(ConstMean(), cov)
            train_gp(gp, X, data_Y, steps=500)
            gpr = PosteriorGP(gp, X, data_Y)
            assert (gpr.mean(X)-Y).var().sqrt() < 0.05
            assert (gpr.grad(X)-dY).var().sqrt() < 0.05


if __name__ == '__main__':
    # test_basic()
    # test_grad()
    test_multidim()

