import torch
from math import pi
from theforce.regression.algebra import free_form, positive


def bordered(m, c, r, d):
    if r is None:
        r = c.T
    mm = torch.cat([torch.cat([m, c], dim=1),
                    torch.cat([r, d], dim=1)])
    return mm


def entropy(a):
    return -(a*a.log()).sum()


class SPD(torch.Tensor):

    def __init__(self, data, lbound=1e-3):
        self.data = data
        self.lbound = lbound
        self._inverse = None

    def inverse(self):
        if self._inverse is None:
            self._inverse = self.data.cholesky().cholesky_inverse()
        return self._inverse

    def append_(self, column, diagonal, lbound=None):
        a = column.view(-1, 1)
        i = torch.as_tensor(diagonal).view(1, 1)
        alpha = self.inverse()@a
        v = (i-a.T@alpha)
        if v < (lbound if lbound else self.lbound):
            return False
        beta = v.sqrt()
        self.data = bordered(self.data, a, a.T, i)
        self._inverse = bordered(self.inverse() + (alpha@alpha.T)/v,
                                 -alpha/v, None, torch.ones(1, 1)/v)
        return True

    def pop_(self, i):
        k = torch.cat([torch.arange(i), torch.arange(i+1, self.size(0))])
        self.data = self.data.index_select(0, k).index_select(1, k)
        alpha = self._inverse[i].index_select(0, k).view(-1, 1)
        beta = self._inverse[i, i]
        self._inverse = (self._inverse.index_select(0, k).index_select(1, k) -
                         alpha@alpha.T/beta)

    def test(self):
        a = (self.data@self.inverse() - torch.eye(self.size(0))).abs().max()
        return a


class CholSPD(torch.Tensor):

    def __init__(self, data, lbound=1e-3):
        self.data = data
        self.lbound = lbound
        self._cholesky = None

    def cholesky(self):
        if self._cholesky is None:
            self._cholesky = self.data.cholesky()
            self._inverse_of_cholesky = self._cholesky.inverse()
            self._inverse = self._cholesky.cholesky_inverse()
        return self._cholesky

    def inverse(self):
        if self._cholesky is None:
            self.cholesky()
        return self._inverse

    def append_(self, column, diagonal, lbound=None):
        a = column.view(-1, 1)
        i = torch.as_tensor(diagonal).view(1, 1)
        alpha = self.inverse()@a
        v = (i-a.T@alpha)
        if v < (lbound if lbound else self.lbound):
            return False
        beta = v.sqrt()
        self.data = bordered(self.data, a, a.T, i)
        self._cholesky = bordered(self._cholesky, torch.zeros_like(a),
                                  (self._inverse_of_cholesky@a).T, beta)
        self._inverse_of_cholesky = bordered(self._inverse_of_cholesky,
                                             torch.zeros_like(a),
                                             -alpha.T/beta, 1./beta)
        self._inverse = bordered(self._inverse + (alpha@alpha.T)/v,
                                 -alpha/v, None, torch.ones(1, 1)/v)
        return True

    def log_prob(self, _y):
        y = torch.as_tensor(_y).view(-1, 1)
        f = -0.5*(y.T@self.inverse()@y + 2*self._cholesky.diag().log().sum() +
                  y.size(0)*torch.log(torch.tensor(2*pi)))
        return f


class Manifold:

    def __init__(self, K, y, backend=CholSPD):
        """
        With backend CholSPD, log_prob becomes available.
        With backend SPD, backward_ (with pop_, swap_) becomes available.
        forward_ is available in any case.
        """
        self.K = backend(K)
        self.y = y.view(-1, 1)
        self._mu = None

    def __call__(self, col):
        return col@self.mu

    @property
    def mu(self):
        if self._mu is None:
            self._mu = self.K.inverse()@self.y
        return self._mu

    def append_(self, col, diag, y):
        if self.K.append_(col, diag):
            self.y = torch.cat([self.y, y.view(1, 1)])
            self._mu = None
            return True
        else:
            return False

    def pop_(self, j):
        self.K.pop_(j)
        self.y = torch.cat([self.y[:j], self.y[j+1:]])
        self._mu = None

    def forward_(self, col, diag, y, diff=None):
        beta = (diag - col.view(1, -1)@self.K.inverse()@col.view(-1, 1)).sqrt()
        delta = self(col)-y
        if delta.abs() > (diff if diff else beta):
            return self.append_(col, diag, y)
        else:
            return False

    def backward_(self, diff=None):
        indices = list(range(self.K.size(0)))
        for j in indices[-1::-1]:
            Lambda = self.K.inverse()[j]
            beta = 1./Lambda[j].sqrt()
            Lambda = (Lambda*beta).view(-1, 1)
            delta = (self.K[j]@Lambda)@(Lambda.T@self.y)
            if delta.abs() < (diff if diff else beta):
                self.pop_(j)
                del indices[j]
        return indices

    def swap_(self, _col, diag, y, j=None):
        col = _col.view(-1)
        # closest to col
        if j is None:
            j = col.argmax()
        # inverse without j'th col
        lam1 = self.K.inverse()[j]
        beta = 1./lam1[j].sqrt()
        lam1 = (lam1*beta).view(-1, 1)
        Ki = self.K.inverse() - lam1@lam1.T
        # inverse with new col
        alpha = Ki@col
        beta = (diag - col@alpha).sqrt()
        if beta**2 < self.K.lbound:
            return None
        alpha[j] = -1.
        lam2 = (alpha/beta).view(-1, 1)
        Ki = Ki + lam2@lam2.T
        # old y with altered model
        mu = Ki@self.y+Ki[:, j].view(-1, 1)*(y-self.y[j])
        yj = self.K[j].view(1, -1)@mu + (col[j]-self.K[j, j])*mu[j]
        # swap
        del1 = self(col)-y
        del2 = yj-self.y[j]
        if del2.abs() < del1.abs():
            self.K[j] = self.K[:, j] = col
            self.K[j, j] = diag
            self.y[j] = y
            self.K._inverse = Ki
            self._mu = mu
            return j
        else:
            return None

    def log_prob(self):
        return self.K.log_prob(self.y)


class Model:

    def __init__(self, kern, cat=True, signal=None, force_norm=False):
        self.kern = kern
        self.x = []
        self.man = None
        self.cat = cat
        self.params = kern.params
        self.signal = signal
        self.force_norm = force_norm

    @property
    def signal(self):
        return 1. if self._signal is None else positive(self._signal)

    @signal.setter
    def signal(self, value):
        if value is None:
            self._signal = None
        else:
            v = torch.as_tensor(value)
            assert v > 0
            self._signal = Parameter(free_form(v))
            self.params.append(self._signal)

    def forward_(self, x, y, diff=None):
        alpha = self.kern(x, x)
        norm = alpha if self.force_norm else 1.
        if self.man is None:
            self.man = Manifold(alpha*self.signal/norm, y)
            self.x += [x]
            return True
        elif self.man.forward_(self.kern(x, torch.cat(self.x) if self.cat else self.x)*self.signal/norm,
                               alpha*self.signal/norm, y, diff=diff):
            self.x += [x]
            return True
        else:
            return False

    def log_prob(self, remake=True, diff=None):
        if remake:
            x = self.x
            y = self.man.y
            self.x = []
            self.man = None
            for e, f in zip(*[x, y]):
                self.forward_(e, f, diff=diff)
        return self.man.log_prob()

    def opt_step(self, opt, diff=None):
        opt.zero_grad()
        loss = -self.log_prob(diff=diff)
        loss.backward()
        opt.step()
        return loss

    def optimize(self, opt, delta=None, maxsteps=1000, diff=None):
        _loss = self.opt_step(opt, diff=diff)
        for step in range(maxsteps-1):
            loss = self.opt_step(opt, diff=diff)
            if delta and (loss-_loss).abs() < delta:
                break
            _loss = loss
        return step+1, loss


def test_spd(self):
    a = (self.data@self.inverse() - torch.eye(self.size(0))).abs().max()
    b = (self._cholesky@self._inverse_of_cholesky -
         torch.eye(self.size(0))).abs().max()
    c = (self.data - self._cholesky@self._cholesky.T).abs().max()
    return a, b, c


def test_rbf():
    from theforce.regression.stationary import RBF
    kern = RBF(signal=1.)
    x = torch.randn(1)
    K = SPD(kern(x, x))
    for _ in range(1000):
        xx = torch.randn(1)
        a = kern(xx, x)
        if K.append_(a, 1.):
            x = torch.cat([x, xx])
    K.pop_(3)
    K.pop_(0)
    print(K.test())


if __name__ == '__main__':
    test_rbf()
