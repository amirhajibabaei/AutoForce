
# coding: utf-8

# In[ ]:


from theforce.similarity.similarity import SimilarityKernel
from theforce.regression.algebra import positive, free_form
from torch import zeros, cat, stack
from theforce.util.util import iterable
import torch
from torch.nn import Module, Parameter


class PairSimilarityKernel(SimilarityKernel):

    def __init__(self, kernels, a, b):
        super().__init__([kern for kern in iterable(kernels)])
        self.a = a
        self.b = b

    @property
    def state_args(self):
        return super().state_args + ', {}, {}'.format(self.a, self.b)

    def descriptor(self, r):
        raise NotImplementedError(
            'descriptor shoud be implemented in a child class')

    def save_for_later(self, loc, keyvals):
        for key, val in keyvals.items():
            setattr(loc, self.name+'_'+key, val)

    def saved(self, atoms_or_loc, key):
        return getattr(atoms_or_loc, self.name+'_'+key)

    def precalculate(self, loc):
        loc.select(self.a, self.b, bothways=True)
        d, grad = self.descriptor(loc.r)
        data = {'diag_value': d, 'diag_grad': grad}
        self.save_for_later(loc, data)
        m = (loc.j > loc.i)
        if self.a == self.b:
            m = m | ((loc.j == loc.i) & loc.lex)
        data = {'value': d[m], 'grad': grad[m], 'i': loc.i[m], 'j': loc.j[m]}
        self.save_for_later(loc, data)
        if hasattr(self, 'factor'):
            fac, facgrad = self.factor(d)
            self.save_for_later(loc, {'diag_fac': fac, 'diag_facgrad': facgrad,
                                      'fac': fac[m], 'facgrad': facgrad[m]})
            self.has_factor = True
            if hasattr(self.factor, 'state'):
                self.has_state = True
                self.save_for_later(loc, {'m': m, 'state': self.factor.state})
            else:
                self.has_state = False
        else:
            self.has_factor = False

    def recalculate(self, atoms_or_loc):
        if self.has_factor:
            if self.has_state:
                state = self.factor.state
                for loc in iterable(atoms_or_loc):
                    if state != self.saved(loc, 'state'):
                        d = self.saved(loc, 'diag_value')
                        m = self.saved(loc, 'm')
                        fac, facgrad = self.factor(d)
                        self.save_for_later(loc, {'diag_fac': fac, 'diag_facgrad': facgrad,
                                                  'fac': fac[m], 'facgrad': facgrad[m],
                                                  'state': self.factor.state})

    def func(self, p, q):
        d = self.saved(p, 'value')
        dd = self.saved(q, 'value')
        c = self.kern(d, dd)
        if self.has_factor:
            self.recalculate(p)
            self.recalculate(q)
            f = self.saved(p, 'fac')
            ff = self.saved(q, 'fac')
            c = c * (f*ff.t())
        return c.sum().view(1, 1)

    def leftgrad(self, p, q):
        d = self.saved(p, 'value')
        grad = self.saved(p, 'grad')
        i = self.saved(p, 'i')
        j = self.saved(p, 'j')
        dd = self.saved(q, 'value')
        c = self.kern.leftgrad(d, dd)
        if self.has_factor:
            self.recalculate(p)
            self.recalculate(q)
            f = self.saved(p, 'fac')
            fg = self.saved(p, 'facgrad')
            ff = self.saved(q, 'fac')
            c = c*(f*ff.t()) + (fg*ff.t())*self.kern(d, dd)
        c = c[:, None] * grad[..., None]
        c = c.sum(dim=-1).view(-1, 3)
        g = zeros(p.natoms, 3).index_add(0, i, -c).index_add(0, j, c)
        return g.view(-1, 1)

    def rightgrad(self, p, q):
        d = self.saved(p, 'value')
        dd = self.saved(q, 'value')
        grad = self.saved(q, 'grad')
        i = self.saved(q, 'i')
        j = self.saved(q, 'j')
        c = self.kern.rightgrad(d, dd)
        if self.has_factor:
            self.recalculate(p)
            self.recalculate(q)
            f = self.saved(p, 'fac')
            ff = self.saved(q, 'fac')
            ffg = self.saved(q, 'facgrad')
            c = c*(f*ff.t()) + (f*ffg.t())*self.kern(d, dd)
        c = c[..., None] * grad
        c = c.sum(dim=0).view(-1, 3)
        g = zeros(q.natoms, 3).index_add(0, i, -c).index_add(0, j, c)
        return g.view(1, -1)

    def gradgrad(self, p, q):
        raise NotImplementedError('Not defined yet')

    def gradgraddiag(self, p):
        forces = []
        for i, loc in enumerate(iterable(p.loc)):
            d = self.saved(loc, 'diag_value')
            grad = self.saved(loc, 'diag_grad')
            c = self.kern.gradgrad(d, d)
            if self.has_factor:
                self.recalculate(loc)
                f = self.saved(loc, 'diag_fac')
                fg = self.saved(loc, 'diag_facgrad')
                c = (c*(f*f.t()) + fg*fg.t()*self.kern(d, d) +
                     (fg*f.t()*self.kern.rightgrad(d, d)) +
                     (f*fg.t()*self.kern.leftgrad(d, d)))
            c = c[..., None] * grad[None, ] * grad[:, None]
            c = c.sum(dim=(0, 1))
            forces += [c]
        forces = cat(forces)
        return forces.view(-1)


class DistanceKernel(PairSimilarityKernel):
    def __init__(self, *args):
        super().__init__(*args)

    def descriptor(self, r):
        d = (r**2).sum(dim=-1).sqrt().view(-1, 1)
        grad = r/d
        return d, grad


class LogDistanceKernel(PairSimilarityKernel):
    def __init__(self, *args):
        super().__init__(*args)

    def descriptor(self, r):
        d = (r**2).sum(dim=-1).sqrt().view(-1, 1)
        grad = r/d**2
        return d.log(), grad


class RepulsiveCoreKernel(DistanceKernel):
    def __init__(self, *args, eta=1):
        super().__init__(*args)
        self.eta = eta

    def factor(self, d):
        return 1./d**self.eta, -self.eta/d**(self.eta+1)

    @property
    def state_args(self):
        return super().state_args + ', eta={}'.format(self.eta)


class PairKernel(DistanceKernel):
    def __init__(self, *args, factor=None):
        super().__init__(*args)
        if factor is not None:
            self.factor = factor
            if hasattr(factor, 'params'):
                self.params += factor.params

    @property
    def state_args(self):
        return super().state_args + ', factor={}'.format(
            self.factor.state if hasattr(self, 'factor')
            and hasattr(self.factor, 'state') else None)


class PairCut(Module):

    def __init__(self, cutoff):
        super().__init__()
        self.rc = cutoff

    @property
    def state_args(self):
        return '{}'.format(self.rc)

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

    def forward(self, d):
        step = torch.where(d < self.rc, torch.ones_like(d),
                           torch.zeros_like(d))
        f, g = self.func_and_grad(d)
        return step*f, step*g

    def func_and_grad(self, d):
        raise NotImplementedError('func_and_grad')


class PolyCut(PairCut):

    def __init__(self, cutoff, n=2):
        super().__init__(cutoff)
        self.n = n

    def func_and_grad(self, d):
        return (d-self.rc)**self.n, self.n*(d-self.rc)**(self.n-1)

    @property
    def state_args(self):
        return super().state_args + ', n={}'.format(self.n)


class RepulsiveCore(Module):

    def __init__(self, eta=1):
        super().__init__()
        self.eta = eta

    def forward(self, d):
        return 1./d**self.eta, -self.eta/d**(self.eta+1)

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

    def forward(self, d):
        return self.z/d**self.eta, -self.eta*self.z/d**(self.eta+1)

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

    def forward(self, d):
        f, df = self.f(d)
        g, dg = self.g(d)
        return f*g, df*g + f*dg

    @property
    def state_args(self):
        return '{}, {}'.format(self.f.state, self.g.state)

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)


def example():
    from torch import tensor
    from theforce.regression.core import SquaredExp

    factor = Product(PolyCut(1.0), ParamedRepulsiveCore())
    kern = PairKernel(SquaredExp(), 1, 1, factor=factor)
    k = eval(kern.state)
    d = torch.arange(0.1, 1.5, 0.1).view(-1)
    d.requires_grad = True
    f = eval(factor.state)
    a, b = f(d)
    a.sum().backward()
    print(d.grad.allclose(b))


def example_optim():
    cut = 1.63
    d = torch.linspace(0.1, cut*1.3, 50).view(-1, 1)
    Y = (3.7/d**2.4)*(d-cut)**2
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
    example()
    example_optim()

