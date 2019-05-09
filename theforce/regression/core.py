
# coding: utf-8

# In[ ]:


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


class PNorm(Module):

    def __init__(self, p=2, **kwargs):
        super().__init__()
        self.r = Displacement(**kwargs)
        self.p = p

    def forward(self, true_norm=False, **kwargs):
        norm = (self.r(**kwargs).abs().pow(self.p)).sum(dim=-1)
        if true_norm:
            norm = norm**(1.0/self.p)
        return norm


class Stationary(Module):
    """ [p=2, dim=1] """

    def __init__(self, **kwargs):
        super().__init__()
        self.rp = PNorm(**kwargs)
        self._signal = Parameter(free_form(torch.ones(1)))

    def forward(self, **kwargs):
        return positive(self._signal).pow(2)*self.cov_func(self.rp(**kwargs))

    def cov_func(self, rp):
        raise NotImplementedError(
            'This should be implemented in a child class!')


class SquaredExp(Stationary):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cov_func(self, rp):
        return (-rp/2).exp()


def test():
    if 1:
        r = PNorm(dim=1, p=torch.rand(1))
        x = torch.tensor([[0.0], [1.]])
        assert (r(x=x, true_norm=True) == r(xx=x, true_norm=True)).all()
        assert (r(x=x, true_norm=True) == torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]])).all()

    if 1:
        dim = 7
        kern = SquaredExp(p=2, dim=dim)
        x = torch.rand(19, dim)
        xx = torch.rand(37, dim)
        K = kern(x=x, xx=xx)
        assert torch.allclose(K, (-(x[:, None]-xx[None])**2/2)
                              .sum(dim=-1).exp())


if __name__ == '__main__':
    test()

