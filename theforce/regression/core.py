
# coding: utf-8

# In[1]:


import torch
from torch.nn import Module, Parameter
from theforce.regression.algebra import free_form, positive


class Displacement(Module):

    def __init__(self, dim=1):
        super().__init__()
        self._scale = Parameter(free_form(torch.ones(dim)))

    def forward(self, x=None, xx=None):
        if xx is None and x is None:
            return torch.ones(0, 0, self._scale.size(0))
        elif x is None:
            x = xx
        elif xx is None:
            xx = x
        return (x[:, None]-xx[None])/positive(self._scale)

    def extra_repr(self):
        print('length scales: {}'.format(positive(self._scale)))


class PNorm(Module):

    def __init__(self, power=2, **kwargs):
        super().__init__()
        self.r = Displacement(**kwargs)
        self.p = power

    def forward(self, true_norm=False, **kwargs):
        norm = (self.r(**kwargs).abs().pow(self.p)).sum(dim=-1)
        if true_norm:
            norm = norm**(1.0/self.p)
        return norm


class Stationary(Module):
    """ [p=2, dim=1] """

    def __init__(self, signal=torch.ones(1), **kwargs):
        super().__init__()
        self.rp = PNorm(**kwargs)
        self._signal = Parameter(free_form(torch.as_tensor(signal)))

    def forward(self, x=None, xx=None):
        return positive(self._signal).pow(2)*self.cov_func(self.rp(x=x, xx=xx))

    def cov_func(self, rp):
        """ if r=0 it should return 1 """
        raise NotImplementedError(
            'This should be implemented in a child class!')

    def diag(self, x=None):
        if x is None:
            return positive(self._signal).pow(2)
        else:
            return positive(self._signal).pow(2)*torch.ones(x.size(0))

    def extra_repr(self):
        print('signal variance: {}'.format(positive(self._signal).pow(2)))


class SquaredExp(Stationary):

    def __init__(self, **kwargs):
        super().__init__(power=2, **kwargs)

    def cov_func(self, rp):
        return (-rp/2).exp()


class White(Stationary):

    def __init__(self, signal=1e-3, **kwargs):
        super().__init__(signal=signal, **kwargs)

    def forward(self, x=None, xx=None):
        x_in = x is not None
        xx_in = xx is not None
        if not x_in and not xx_in:
            return self.diag()
        elif x_in and not xx_in:
            return self.diag(x).diag()
        elif xx_in and not x_in:
            return self.diag(xx).diag()
        elif x_in and xx_in:
            if x.shape == xx.shape and torch.allclose(x, xx):
                return self.diag(x).diag()
            else:
                return torch.zeros(x.size(0), xx.size(0))


def test():
    if 1:
        r = PNorm(dim=1, power=torch.rand(1))
        x = torch.tensor([[0.0], [1.]])
        assert (r(x=x, true_norm=True) == r(xx=x, true_norm=True)).all()
        assert (r(x=x, true_norm=True) == torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]])).all()

    if 1:
        dim = 7
        kern = SquaredExp(dim=dim)
        x = torch.rand(19, dim)
        xx = torch.rand(37, dim)
        K = kern(x=x, xx=xx)
        assert torch.allclose(K, (-(x[:, None]-xx[None])**2/2)
                              .sum(dim=-1).exp())

    if 1:
        white = White(signal=1.0)
        x = torch.rand(7, 1)
        assert (white(x, x) == torch.eye(7)).all()


if __name__ == '__main__':
    test()

