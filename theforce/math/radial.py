
import torch
from torch.nn import Module, Parameter
from theforce.regression.algebra import free_form, positive


class RepulsiveCore(Module):

    def __init__(self, eta=1):
        super().__init__()
        self.eta = eta

    def forward(self, d, grad=True):
        if grad:
            return 1./d**self.eta, -self.eta/d**(self.eta+1)
        else:
            return 1./d**self.eta

    @property
    def state_args(self):
        return 'eta={}'.format(self.eta)

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)


class ParamedRepulsiveCore(Module):

    def __init__(self, z=1.0, lb=1.0, beta=1e-6):
        """z/r**eta where eta=lb+beta and beta>0"""
        super().__init__()
        assert beta > 0
        self.z = z
        self.lb = lb
        self.beta = beta

    def forward(self, d, grad=True):
        if grad:
            return self.z/d**self.eta, -self.eta*self.z/d**(self.eta+1)
        else:
            return self.z/d**self.eta

    @property
    def eta(self):
        return self.lb + self.beta

    @property
    def beta(self):
        return positive(self._beta)

    @beta.setter
    def beta(self, value):
        self._beta = Parameter(free_form(torch.as_tensor(value)))

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        self._z = Parameter(torch.as_tensor(value))

    @property
    def params(self):
        return [self._z, self._beta]

    @property
    def state_args(self):
        return 'z={}, lb={}, beta={}'.format(self.z, self.lb, self.beta)

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)


class Product(Module):

    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g
        self.params = []
        for a in [f, g]:
            if hasattr(a, 'params'):
                self.params += a.params

    def forward(self, d, grad=True):
        f = self.f(d, grad=grad)
        g = self.g(d, grad=grad)
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
        return self.__class__.__name__+'({})'.format(self.state_args)


def example_optim():
    from theforce.math.cutoff import PolyCut
    cut = 1.63
    d = torch.linspace(0.1, cut*1.3, 50).view(-1, 1)
    Y = (3.7/d**2.4)*(1-d/cut)**2
    fac = Product(ParamedRepulsiveCore(), PolyCut(cut))
    optimizer = torch.optim.Adam([{'params': fac.params}], lr=0.5)

    for _ in range(1000):
        optimizer.zero_grad()
        a, b = fac(d)
        loss = ((a-Y)**2).sum()
        loss.backward()
        optimizer.step()

    print(fac.state)
    print(fac.__class__.__name__)


if __name__ == '__main__':
    example_optim()

