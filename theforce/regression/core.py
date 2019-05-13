
# coding: utf-8

# In[ ]:


""" experimental """
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

    def delta(self):
        return torch.eye(self._scale.size(0))

    def divide(self, operation):
        if operation is 'func':
            return 1.0
        else:
            scale = positive(self._scale)
            if operation is 'grad':
                return scale
            elif operation is 'hessian':
                return scale[None]*scale[:, None]

    def extra_repr(self):
        print('length scales: {}'.format(positive(self._scale)))


class Stationary(Module):
    """ [dim=1, signal=1] """

    def __init__(self, dim=1, signal=1.0):
        super().__init__()
        self.r = Displacement(dim=dim)
        self._signal = Parameter(free_form(torch.as_tensor(signal)))
        self.params = [self.r._scale, self._signal]

    def forward(self, x=None, xx=None, operation='func'):
        return positive(self._signal).pow(2)*getattr(self, operation)(self.r(x=x, xx=xx))             / self.r.divide(operation)

    def diag(self, x=None):
        if x is None:
            return positive(self._signal).pow(2)
        else:
            return positive(self._signal).pow(2)*torch.ones(x.size(0))

    def extra_repr(self):
        print('signal variance: {}'.format(positive(self._signal).pow(2)))


class SquaredExp(Stationary):
    """ [dim=1, signal=1] """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def func(self, r):
        return (-(r**2).sum(dim=-1)/2).exp()

    def grad(self, r):
        cov = self.func(r)
        return -r*cov[..., None]

    def hessian(self, r):
        cov = self.func(r)
        return (r[..., None, :]*r[..., None]-self.r.delta())*cov[..., None, None]


class White(Stationary):

    def __init__(self, signal=1e-3, **kwargs):
        super().__init__(signal=signal, **kwargs)
        self._signal.requires_grad = False

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
        dim = 7
        kern = SquaredExp(dim=dim)
        x = torch.rand(19, dim)
        xx = torch.rand(37, dim)
        K = kern(x=x, xx=xx)
        assert torch.allclose(K, (-(x[:, None]-xx[None])**2/2)
                              .sum(dim=-1).exp())
        print(kern(x, xx, 'func').shape)
        print(kern(x, xx, 'grad').shape)
        print(kern(x, xx, 'hessian').shape)

    if 1:
        white = White(signal=1.0)
        x = torch.rand(7, 1)
        assert (white(x, x) == torch.eye(7)).all()


if __name__ == '__main__':
    test()

