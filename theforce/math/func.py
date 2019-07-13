
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module


def _positive(x):
    return torch.log(1. + torch.exp(x))


def _free(x):
    return torch.log(torch.exp(x) - 1.)


class Func(Module):

    def __init__(self):
        super().__init__()
        self.params = []

    @property
    def state_args(self):
        return ''

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

    def __repr__(self):
        return self.state

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def __pow__(self, n):
        return Pow(f=self, n=n)


class I(Func):
    """Identity"""

    def __init__(self):
        super().__init__()

    def forward(self, x, grad=True):
        if grad:
            return x, torch.ones_like(x)
        else:
            return x

    @property
    def state_args(self):
        return ''


class Add(Func):

    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g
        self.params = f.params + g.params

    def forward(self, x, grad=True):
        f = self.f(x, grad=grad)
        g = self.g(x, grad=grad)
        if grad:
            f, df = f
            g, dg = g
            return f+g, df+dg
        else:
            return f+g

    @property
    def state_args(self):
        return '{}, {}'.format(self.f.state, self.g.state)

    @property
    def state(self):
        return '({}+{})'.format(self.f.state, self.g.state)


class Sub(Func):

    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g
        self.params = f.params + g.params

    def forward(self, x, grad=True):
        f = self.f(x, grad=grad)
        g = self.g(x, grad=grad)
        if grad:
            f, df = f
            g, dg = g
            return f-g, df-dg
        else:
            return f-g

    @property
    def state_args(self):
        return '{}, {}'.format(self.f.state, self.g.state)

    @property
    def state(self):
        return '({}-{})'.format(self.f.state, self.g.state)


class Mul(Func):

    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g
        self.params = f.params + g.params

    def forward(self, x, grad=True):
        f = self.f(x, grad=grad)
        g = self.g(x, grad=grad)
        if grad:
            f, df = f
            g, dg = g
            return f*g, df*g + f*dg
        else:
            return f*g

    @property
    def state_args(self):
        return '{}, {}'.format(self.f.state, self.g.state)

    @property
    def state(self):
        return '{}*{}'.format(self.f.state, self.g.state)


class Div(Func):

    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g
        self.params = f.params + g.params

    def forward(self, x, grad=True):
        f = self.f(x, grad=grad)
        g = self.g(x, grad=grad)
        if grad:
            f, df = f
            g, dg = g
            return f/g, (df*g - f*dg)/g**2
        else:
            return f/g

    @property
    def state_args(self):
        return '{}, {}'.format(self.f.state, self.g.state)

    @property
    def state(self):
        return '{}/{}'.format(self.f.state, self.g.state)


class Param:
    """example: Param(Real, 1.0, 'x')"""
    _params = {}

    def __new__(cls, _class, r, name, rg=True):
        try:
            return Param._params[name]
        except KeyError:
            c = _class(r, rg=rg, name=name)
            Param._params[name] = c
            return c
# In Real, Positive, and Negative the "name" kw is included for the benefit pf Param.


class Real(Func):

    def __init__(self, r=1e-6, rg=False, name=None):
        super().__init__()
        self.r = torch.as_tensor(r)
        self.r.requires_grad = rg
        self.params = [self.r]
        self.name = name

    def forward(self, x, grad=True):
        if grad:
            return self.r, 0
        else:
            return self.r

    @property
    def state_args(self):
        return '{}, rg={}'.format(self.r.data, self.r.requires_grad)

    @property
    def state(self):
        if self.name:
            return "Param({}, {}, name='{}')".format(self.__class__.__name__, self.state_args, self.name)
        else:
            return super().state


class Positive(Func):

    def __init__(self, r=1.0, rg=False, name=None):
        super().__init__()
        assert r > 0.0
        self._r = _free(torch.as_tensor(r))
        self._r.requires_grad = rg
        self.params = [self._r]
        self.name = name

    def forward(self, x, grad=True):
        if grad:
            return _positive(self._r), 0
        else:
            return _positive(self._r)

    @property
    def state_args(self):
        return '{}, rg={}'.format(_positive(self._r.data), self._r.requires_grad)

    @property
    def state(self):
        if self.name:
            return "Param({}, {}, name='{}')".format(self.__class__.__name__, self.state_args, self.name)
        else:
            return super().state


class Negative(Func):

    def __init__(self, r=-1.0, rg=False, name=None):
        super().__init__()
        assert r < 0.0
        self._r = _free(torch.as_tensor(-r))
        self._r.requires_grad = rg
        self.params = [self._r]
        self.name = name

    def forward(self, x, grad=True):
        if grad:
            return -_positive(self._r), 0
        else:
            return -_positive(self._r)

    @property
    def state_args(self):
        return '{}, rg={}'.format(-_positive(self._r.data), self._r.requires_grad)

    @property
    def state(self):
        if self.name:
            return "Param({}, {}, name='{}')".format(self.__class__.__name__, self.state_args, self.name)
        else:
            return super().state


class Pow(Func):  # TODO: Func**Func

    def __init__(self, f=I(), n=1):
        super().__init__()
        self.f = f
        self.n = n
        self.params = f.params

    def forward(self, x, grad=True):
        f = self.f(x, grad=grad)
        if grad:
            f, g = f
            return f**self.n, self.n*g*f**(self.n-1)
        else:
            return f**self.n

    @property
    def state_args(self):
        return 'f={}, n={}'.format(self.f.state, self.n)

    @property
    def state(self):
        if self.n < 0:
            return '{}**({})'.format(self.f.state, self.n)
        else:
            return '{}**{}'.format(self.f.state, self.n)


class Exp(Func):

    def __init__(self, f=I()):
        super().__init__()
        self.f = f
        self.params = f.params

    def forward(self, x, grad=True):
        f = self.f(x, grad=grad)
        if grad:
            f, g = f
            y = f.exp()
            return y, g*y
        else:
            return f.exp()

    @property
    def state_args(self):
        return 'f={}'.format(self.f.state)


def test():
    f = Exp((I()-Real(1.0))**2/Negative(-1/3))
    x = torch.arange(-1, 3, 0.1, requires_grad=True)
    a, b = f(x)
    a.sum().backward()
    print(x.grad.allclose(b))
    print(eval(f.state).state == f.state)

    d = Real(1.)
    print(id(eval(d.state)) != id(d))

    a = Param(Real, 1.0, 'x')
    b = Param(Positive, 1.0, 'y')
    c = Param(Negative, -1.0, 'z')
    print([id(eval(v.state)) == id(v) for v in (a, b, c)])


if __name__ == '__main__':
    test()

