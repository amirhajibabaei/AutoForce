
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module, Parameter
from theforce.regression.algebra import positive, free_form


def atleast2d(t, force2d=False):
    _t = torch.as_tensor(t)
    if _t.dim() < 2:
        return _t.view(-1, 1)
    elif _t.dim() >= 2:
        if force2d:
            return _t.view(_t.size(0), torch.tensor(_t.size()[1:]).prod())
        else:
            return _t


class Kernel(Module):
    """Kernel is a function that accepts two arguments."""

    def __init__(self):
        super().__init__()
        self.params = []

    def checkout_inputs(self, x, xx=None, diag=False):
        t = atleast2d(x, force2d=False)
        s = t.size()[1:]  # features original shape
        t = atleast2d(t, force2d=True).t().contiguous()
        if diag:
            assert xx is None
            return t, t, s
        else:
            if xx is None:
                tt = t
            else:
                tt = atleast2d(xx, force2d=True).t().contiguous()
            t, tt = torch.broadcast_tensors(t[..., None], tt[:, None])
            return t, tt, s

    # main mathods
    def forward(self, x, xx=None, diag=False, method='func'):
        t, tt, s = self.checkout_inputs(x, xx, diag)
        k = getattr(self, 'get_'+method)(t, tt)
        if k.size(-1) == 1 or (not diag and k.size(-2) == 1):
            k = torch.ones_like(t[0])*k  # TODO: use .expand instead
        return k

    def func(self, x, xx=None, diag=False):
        return self.forward(x, xx, diag, method='func')

    def leftgrad(self, x, xx=None, diag=False):
        return self.forward(x, xx, diag, method='leftgrad')

    def rightgrad(self, x, xx=None, diag=False):
        return self.forward(x, xx, diag, method='rightgrad')

    def gradgrad(self, x, xx=None, diag=False):
        return self.forward(x, xx, diag, method='gradgrad')

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

    # Operators
    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __pow__(self, other):
        return Pow(self, other)

    def pow(self, eta):
        return Pow(self, eta)

    def exp(self):
        return Exp(self)

    # overload the following methods
    @property
    def state_args(self):
        return ''

    def get_func(self, x, xx):
        """output shape: (m, n)"""
        raise NotImplementedError(
            'get_func in {}'.format(self.__class__.__name__))

    def get_leftgrad(self, x, xx):
        """output shape: (d, m, n)"""
        raise NotImplementedError(
            'get_leftgrad in {}'.format(self.__class__.__name__))

    def get_rightgrad(self, x, xx):
        """output shape: (d, m, n)"""
        raise NotImplementedError(
            'get_rightgrad in {}'.format(self.__class__.__name__))

    def get_gradgrad(self, x, xx):
        """output shape: (d, d, m, n)"""
        raise NotImplementedError(
            'get_gradgrad in {}'.format(self.__class__.__name__))


class BinaryOperator(Kernel):

    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.params = a.params + b.params

    @property
    def state_args(self):
        return '{}, {}'.format(self.a.state, self.b.state)


class Add(BinaryOperator):

    def __init__(self, *args):
        super().__init__(*args)

    def get_func(self, x, xx):
        return self.a.get_func(x, xx) + self.b.get_func(x, xx)

    def get_leftgrad(self, x, xx):
        k = self.a.get_leftgrad(x, xx) + self.b.get_leftgrad(x, xx)
        return k

    def get_rightgrad(self, x, xx):
        k = self.a.get_rightgrad(x, xx) + self.b.get_rightgrad(x, xx)
        return k

    def get_gradgrad(self, x, xx):
        k = self.a.get_gradgrad(x, xx) + self.b.get_gradgrad(x, xx)
        return k


class Sub(BinaryOperator):

    def __init__(self, *args):
        super().__init__(*args)

    def get_func(self, x, xx):
        return self.a.get_func(x, xx) - self.b.get_func(x, xx)

    def get_leftgrad(self, x, xx):
        k = self.a.get_leftgrad(x, xx) - self.b.get_leftgrad(x, xx)
        return k

    def get_rightgrad(self, x, xx):
        k = self.a.get_rightgrad(x, xx) - self.b.get_rightgrad(x, xx)
        return k

    def get_gradgrad(self, x, xx):
        k = self.a.get_gradgrad(x, xx) - self.b.get_gradgrad(x, xx)
        return k


class Mul(BinaryOperator):

    def __init__(self, *args):
        super().__init__(*args)

    def get_func(self, x, xx):
        return self.a.get_func(x, xx)*self.b.get_func(x, xx)

    def get_leftgrad(self, x, xx):
        k = (self.a.get_func(x, xx)*self.b.get_leftgrad(x, xx) +
             self.b.get_func(x, xx)*self.a.get_leftgrad(x, xx))
        return k

    def get_rightgrad(self, x, xx):
        k = (self.a.get_func(x, xx)*self.b.get_rightgrad(x, xx) +
             self.b.get_func(x, xx)*self.a.get_rightgrad(x, xx))
        return k

    def get_gradgrad(self, x, xx):
        k = (self.a.get_func(x, xx)*self.b.get_gradgrad(x, xx) +
             self.b.get_func(x, xx)*self.a.get_gradgrad(x, xx) +
             self.b.get_leftgrad(x, xx)[:, None]*self.a.get_rightgrad(x, xx)[None] +
             self.a.get_leftgrad(x, xx)[:, None]*self.b.get_rightgrad(x, xx)[None])
        return k


class Pow(Kernel):

    def __init__(self, kern, eta):
        super().__init__()
        self.kern = kern
        self.eta = eta

    @property
    def state_args(self):
        return '{}, {}'.format(self.kern.state, self.eta)

    def get_func(self, x, xx):
        return self.kern.get_func(x, xx)**self.eta

    def get_leftgrad(self, x, xx):
        k = (self.eta*self.kern.get_func(x, xx)**(self.eta-1) *
             self.kern.get_leftgrad(x, xx))
        return k

    def get_rightgrad(self, x, xx):
        k = (self.eta*self.kern.get_func(x, xx)**(self.eta-1) *
             self.kern.get_rightgrad(x, xx))
        return k

    def get_gradgrad(self, x, xx):
        k = (self.eta*self.kern.get_func(x, xx)**(self.eta-1)*self.kern.get_gradgrad(x, xx) +
             self.eta*(self.eta-1)*self.kern.get_func(x, xx)**(self.eta-2) *
             self.kern.get_leftgrad(x, xx)[:, None]*self.kern.get_rightgrad(x, xx)[None])
        return k


class Exp(Kernel):
    def __init__(self, kern):
        super().__init__()
        self.kern = kern

    @property
    def state_args(self):
        return '{}'.format(self.kern.state)

    def get_func(self, x, xx):
        return self.kern.get_func(x, xx).exp()

    def get_leftgrad(self, x, xx):
        k = self.kern.get_leftgrad(x, xx)*self.kern.get_func(x, xx).exp()
        return k

    def get_rightgrad(self, x, xx):
        k = self.kern.get_rightgrad(x, xx)*self.kern.get_func(x, xx).exp()
        return k

    def get_gradgrad(self, x, xx):
        k = (self.kern.get_gradgrad(x, xx) + self.kern.get_leftgrad(x, xx)[:, None] *
             self.kern.get_rightgrad(x, xx)[None])*self.kern.get_func(x, xx).exp()
        return k


class Real(Kernel):

    def __init__(self, value):
        super().__init__()
        self.value = torch.as_tensor(value)

    @property
    def state_args(self):
        return '{}'.format(self.value)

    def get_func(self, x, xx):
        return self.value.view((x.dim()-1)*[1])

    def get_leftgrad(self, x, xx):
        return torch.zeros(x.dim()*[1])

    def get_rightgrad(self, x, xx):
        return torch.zeros(x.dim()*[1])

    def get_gradgrad(self, x, xx):
        return torch.zeros((x.dim()+1)*[1])


class Positive(Kernel):

    def __init__(self, signal=1.0, requires_grad=False):
        super().__init__()
        self.signal = signal
        self.requires_grad = requires_grad

    @property
    def signal(self):
        return positive(self._signal)

    @signal.setter
    def signal(self, value):
        v = torch.as_tensor(value)
        assert v > 0
        self._signal = Parameter(free_form(v))
        self.params.append(self._signal)

    @property
    def requires_grad(self):
        return self._signal.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._signal.requires_grad = value

    @property
    def state_args(self):
        return 'signal={}, requires_grad={}'.format(self.signal.data, self.requires_grad)

    def get_func(self, x, xx):
        return self.signal.view((x.dim()-1)*[1])

    def get_leftgrad(self, x, xx):
        return torch.zeros(x.dim()*[1])

    def get_rightgrad(self, x, xx):
        return torch.zeros(x.dim()*[1])

    def get_gradgrad(self, x, xx):
        return torch.zeros((x.dim()+1)*[1])


class White(Kernel):

    def __init__(self, signal=1e-3, requires_grad=False):
        super().__init__()
        self.signal = signal
        self.requires_grad = requires_grad

    @property
    def signal(self):
        return positive(self._signal)

    @signal.setter
    def signal(self, value):
        v = torch.as_tensor(value)
        assert v > 0
        self._signal = Parameter(free_form(v))
        self.params.append(self._signal)

    @property
    def requires_grad(self):
        return self._signal.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._signal.requires_grad = value

    @property
    def state_args(self):
        return 'signal={}, requires_grad={}'.format(self.signal.data, self.requires_grad)

    def get_func(self, x, xx):
        return self.signal*x.isclose(xx).all(dim=0).type(x.type())


class SqD(Kernel):

    def __init__(self):
        super().__init__()

    @property
    def state_args(self):
        return ''

    def get_func(self, x, xx):
        return (x-xx).pow(2).sum(dim=0)

    def get_leftgrad(self, x, xx):
        return 2*(x-xx)

    def get_rightgrad(self, x, xx):
        return -2*(x-xx)

    def get_gradgrad(self, x, xx):
        # Note: output.expand(d, d, *trail) may be needed
        d = x.size(0)
        trail = x.size()[1:]
        return -2*torch.eye(d).view(d, d, *(len(trail)*[1]))


class DotProd(Kernel):

    def __init__(self):
        super().__init__()

    @property
    def state_args(self):
        return ''

    def get_func(self, x, xx):
        return (x*xx).sum(dim=0)

    def get_leftgrad(self, x, xx):
        return torch.ones_like(x)*xx

    def get_rightgrad(self, x, xx):
        return x*torch.ones_like(xx)

    def get_gradgrad(self, x, xx):
        # Note: output.expand(d, d, *trail) may be needed
        d = x.size(0)
        trail = x.size()[1:]
        return torch.eye(d).view(d, d, *(len(trail)*[1]))


def example():
    polynomial = Positive(requires_grad=True) *         (DotProd() + Positive(1e-4, requires_grad=True))**2
    squaredexp = (SqD()*Real(-0.5)).exp()


def test():
    from theforce.regression.core import SquaredExp
    from theforce.regression.gp import Covariance

    x = torch.rand(23, 7)
    xx = torch.rand(19, 7)
    old = Covariance(SquaredExp(dim=7))
    new = (SqD()*Real(-0.5)).exp()
    func = old(x, xx).allclose(new(x, xx))
    leftgrad = old.leftgrad(x, xx).allclose(new.leftgrad(
        x, xx).permute(1, 0, 2).reshape(x.numel(), xx.size(0)))
    rightgrad = old.rightgrad(x, xx).allclose(new.rightgrad(
        x, xx).permute(1, 2, 0).reshape(x.size(0), xx.numel()))
    gradgrad = old.gradgrad(x, xx).allclose(new.gradgrad(
        x, xx).permute(2, 0, 3, 1).reshape(x.numel(), xx.numel()))
    print('Squared-Exponential kernel with two different methods: \n{}\n{}\n{}\n{}'.format(
        func, leftgrad, rightgrad, gradgrad))

    # try empty tensor
    x = torch.rand(0, 7)
    new(x, xx)


if __name__ == '__main__':
    test()

