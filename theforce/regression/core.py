# +
import torch
from torch.nn import Module, Parameter

from theforce.regression.algebra import free_form, positive

"""
1. The kernels are defined such that they return derivatives
of Gram matrix wrt r (r=x-x'). To convert them wrt x, x'
usually just a multiplication by -1 maybe needed.
For consistency the gradgrad in LazyWhite is manually
multiplied by -1.
"""


class LazyWhite(Module):
    """special stationary kernel"""

    def __init__(self, dim=1, signal=0.0, requires_grad=False):
        super().__init__()
        self.dim = dim
        self._signal = Parameter(
            free_form(torch.as_tensor(signal)), requires_grad=requires_grad
        )
        self.params = [self._signal]

    def forward(self, x=None, xx=None, operation="func"):
        x_in = x is not None
        xx_in = xx is not None
        if not x_in and not xx_in:
            k = self.diag()
        elif x_in and not xx_in:
            k = self.diag(x).diag()
        elif xx_in and not x_in:
            k = self.diag(xx).diag()
        elif x_in and xx_in:
            if x.shape == xx.shape and torch.allclose(x, xx):
                k = self.diag(x).diag()
            else:
                k = torch.zeros(x.size(0), xx.size(0))
        if k.dim() == 0 or operation == "func":
            return k
        if operation == "grad":
            return k[..., None] * torch.ones(self.dim)
        elif operation == "gradgrad":
            return (k[..., None, None] * torch.eye(self.dim)).permute(0, 2, 1, 3) * (
                -1
            )  # NOTE 1.

    def diag(self, x=None, operation="func"):
        if x is None:
            return positive(self._signal).pow(2)
        else:
            if operation == "func":
                return positive(self._signal).pow(2) * torch.ones(x.size(0))
            elif operation == "grad":
                raise NotImplementedError("This is not supposed to happen!")
            elif operation == "gradgrad":
                return positive(self._signal).pow(2) * torch.ones(x.numel())

    @property
    def signal(self):
        return positive(self._signal)

    @property
    def state(self):
        return (
            self.__class__.__name__
            + """(dim={}, signal={}, requires_grad={})""".format(
                self.dim, self.signal.data, self.signal.requires_grad
            )
        )


class Displacement(Module):
    def __init__(self, dim=1, scale=None):
        super().__init__()
        if scale is None:
            self._scale = Parameter(free_form(torch.ones(dim)))
        else:
            self._scale = Parameter(free_form(scale))

    def x_xx(self, x=None, xx=None):
        if x is None and xx is None:
            x = torch.ones(0, self._scale.size(0))
            xx = torch.ones(0, self._scale.size(0))
        elif x is None:
            x = xx
        elif xx is None:
            xx = x
        return x, xx

    def forward(self, x=None, xx=None):
        x, xx = self.x_xx(x, xx)
        return (x[:, None] - xx[None]) / positive(self._scale)

    def delta(self):
        return torch.eye(self._scale.size(0))

    def divide(self, operation):
        if operation == "func":
            return 1.0
        else:
            scale = positive(self._scale)
            if operation == "grad":
                return scale
            elif operation == "gradgrad":
                return (scale[None] * scale[:, None])[:, None]

    def extra_repr(self):
        print("length scales: {}".format(positive(self._scale)))

    @property
    def scale(self):
        return positive(self._scale)


class Stationary(Module):
    """[dim=1, signal=1]"""

    def __init__(self, dim=1, signal=1.0, scale=None):
        super().__init__()
        self.r = Displacement(dim=dim, scale=scale)
        self._signal = Parameter(free_form(torch.as_tensor(signal)))
        self.params = [self.r._scale, self._signal]

    def forward(self, x=None, xx=None, operation="func"):
        return (
            positive(self._signal).pow(2)
            * getattr(self, operation)(self.r(x=x, xx=xx))
            / self.r.divide(operation)
        )

    def diag(self, x=None, operation="func"):
        if x is None:
            return positive(self._signal).pow(2)
        else:
            if operation == "func":
                return positive(self._signal).pow(2) * torch.ones(x.size(0))
            elif operation == "grad":
                raise NotImplementedError("This is not supposed to happen!")
            elif operation == "gradgrad":
                return (
                    positive(self._signal).pow(2)
                    * self.d2diag()
                    * (1.0 / self.r.divide("grad")) ** 2
                ).repeat(x.size(0))

    def extra_repr(self):
        print("signal variance: {}".format(positive(self._signal).pow(2)))

    @property
    def signal(self):
        return positive(self._signal)

    @property
    def state(self):
        return self.__class__.__name__ + """(signal={}, scale={})""".format(
            self.signal.data, self.r.scale.data
        )


class SquaredExp(Stationary):
    """[dim=1, signal=1]"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def func(self, r):
        return (-(r**2).sum(dim=-1) / 2).exp()

    def grad(self, r):
        cov = self.func(r)
        return -r * cov[..., None]

    def gradgrad(self, r):
        cov = self.func(r)
        return (
            (r[..., None, :] * r[..., None] - self.r.delta()) * cov[..., None, None]
        ).permute(0, 2, 1, 3)

    def d2diag(self):
        """second deriv of kernel wrt r[i] at r[i]=0"""
        return torch.tensor(-1.0)


def test():
    if 1:
        dim = 2
        kern = SquaredExp(dim=dim)
        x = torch.rand(19, dim)
        xx = torch.rand(37, dim)
        K = kern(x=x, xx=xx)
        assert torch.allclose(
            K, (-((x[:, None] - xx[None]) ** 2) / 2).sum(dim=-1).exp()
        )

    if 1:
        white = LazyWhite(signal=1.0)
        x = torch.rand(13, dim)
        assert (white(x, x) == torch.eye(13)).all()

    if 1:
        kern = SquaredExp(dim=dim)
        white = LazyWhite(dim=dim, signal=1.0)
        assert kern(x, xx, "func").shape == white(x, xx, "func").shape
        assert kern(x, xx, "grad").shape == white(x, xx, "grad").shape
        assert kern(x, xx, "gradgrad").shape == white(x, xx, "gradgrad").shape
        K = white(x, operation="gradgrad")
        assert (
            K.reshape(x.numel(), x.numel()) == (-1) * torch.eye(x.numel())
        ).all()  # see NOTE 1. for -1


if __name__ == "__main__":
    test()
