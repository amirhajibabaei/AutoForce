import torch


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
    print(test_spd(K))


if __name__ == '__main__':
    test_rbf()
