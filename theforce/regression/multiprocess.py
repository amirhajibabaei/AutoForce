
import torch
from torch.nn import Module
from torch.distributions import MultivariateNormal


class MultiProcess(Module):

    def __init__(self, gps):
        super().__init__()
        self.gps = (gps if hasattr(gps, '__iter__') else (gps,))
        self.params = [par for gp in self.gps for par in gp.params]

    def forward(self, Xs, op='func'):
        ps = [gp(x, op=op) for gp, x in zip(self.gps, Xs)]
        loc = torch.stack([p.loc for p in ps]).sum(dim=0)
        cov = torch.stack([p.covariance_matrix for p in ps]).sum(dim=0)
        return MultivariateNormal(loc, covariance_matrix=cov)

    def loss(self, Xs, y):
        if y.dim() == 1:
            loss = -self(Xs, op='func').log_prob(y)
        elif y.shape == x.shape:
            loss = -self(Xs, op='grad').log_prob(y.view(-1))
        cov_loss = sum([gp.covariance_loss for gp in self.gps])
        return loss + cov_loss


class PosteriorMGP(Module):

    def __init__(self, gp, X, Y):
        super().__init__()
        self.x = X
        self.gp = gp
        self.dtype = 'grad' if Y.dim() > 1 else 'func'
        p = gp(X, op=self.dtype)
        self.mu = p.precision_matrix @ (Y.reshape(-1)-p.loc)
        self.kw = {('func', 'func'): 'func', ('func', 'grad'): 'rightgrad',
                   ('grad', 'func'): 'leftgrad', ('grad', 'grad'): 'gradgrad'}

    def loc(self, X, i, op='func'):
        gp = self.gp.gps[i]
        mean = gp.mean(X, operation=op)
        cov = gp.cov(X, self.x[i], operation=self.kw[(op, self.dtype)])
        return mean + ((cov @ self.mu).reshape_as(X) if op == 'grad' else cov @ self.mu)


def test():
    from theforce.regression.core import SquaredExp, LazyWhite
    from theforce.regression.gp import GaussianProcess, ConstMean, Covariance, Inducing, train_gp
    import pylab as plt
    import torch
    from math import pi
    _2pi = 2*torch.tensor(pi)
    get_ipython().run_line_magic('matplotlib', 'inline')

    def func1(X):
        return (0.8*(X+_2pi/12)).sin()*2

    def func2(X):
        return (1.0*(X+_2pi/3)).sin()*4

    def func3(X):
        return (1.2*(X+_2pi/2)).sin()*6

    # data
    X = torch.linspace(-_2pi, _2pi, 39).view(-1, 1)
    X1 = X[torch.randperm(X.size(0))]
    X2 = X[torch.randperm(X.size(0))]
    X3 = X[torch.randperm(X.size(0))]
    Y1, Y2, Y3 = (y.view(-1) for y in (func1(X1), func2(X2), func3(X3)))
    Y = Y1 + Y2 + Y3

    # regression
    # gps = [GaussianProcess(ConstMean(), Covariance((SquaredExp(dim=1), LazyWhite(dim=1, signal=1e-2))))
    #       for _ in range(3)]
    Xind = torch.linspace(-_2pi, _2pi, 11).view(-1, 1)
    gps = [GaussianProcess(ConstMean(), Inducing((SquaredExp(dim=1),), Xind))
           for _ in range(3)]
    mgp = MultiProcess(gps)
    train_gp(mgp, (X1, X2, X3), Y, steps=10)
    gpr = PosteriorMGP(mgp, (X1, X2, X3), Y)

    # plot
    funcs = [func1, func2, func3]
    colors = ['red', 'blue', 'green']
    X = torch.linspace(-_2pi, _2pi, 100).view(-1, 1)
    with torch.no_grad():
        def _(a):
            return a.detach().numpy()
        for i in range(3):
            plt.plot(_(X), _(funcs[i](X)), color=colors[i])
            plt.plot(_(X), _(gpr.loc(X.view(-1, 1), i)), ':', color=colors[i])


if __name__ == '__main__':
    test()

