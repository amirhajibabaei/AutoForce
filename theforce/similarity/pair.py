
# coding: utf-8

# In[ ]:


from theforce.similarity.similarity import SimilarityKernel
from theforce.regression.algebra import positive, free_form
from torch import zeros, cat, stack
from theforce.util.util import iterable
import torch
from torch.nn import Module, Parameter


class PairSimilarityKernel(SimilarityKernel):

    def __init__(self, kernel, a, b):
        super().__init__(kernel)
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
        c = self.kern.leftgrad(d, dd).squeeze(0)
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
        c = self.kern.rightgrad(d, dd).squeeze(0)
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
        d1 = self.saved(p, 'value')
        g1 = self.saved(p, 'grad')
        i1 = self.saved(p, 'i')
        j1 = self.saved(p, 'j')
        d2 = self.saved(q, 'value')
        g2 = self.saved(q, 'grad')
        i2 = self.saved(q, 'i')
        j2 = self.saved(q, 'j')
        c = self.kern.gradgrad(d1, d2)
        if self.has_factor:
            self.recalculate(p)
            self.recalculate(q)
            f1 = self.saved(p, 'fac')
            h1 = self.saved(p, 'facgrad')
            f2 = self.saved(q, 'fac')
            h2 = self.saved(q, 'facgrad')
            c = (c*(f1*f2.t()) + h1*h2.t()*self.kern(d1, d2) +
                 (h1*f2.t()*self.kern.rightgrad(d1, d2)) +
                 (f1*h2.t()*self.kern.leftgrad(d1, d2)))
        c = c.squeeze()[:, None, :, None] * g1[..., None, None] * g2
        cc = torch.zeros(p.natoms, 3, j2.size(0), 3).index_add(
            0, j1, c).index_add(0, i1, -c)
        ccc = torch.zeros(p.natoms, 3, q.natoms, 3).index_add(
            2, j2, cc).index_add(2, i2, -cc)
        return ccc.view(p.natoms*3, q.natoms*3)

    def gradgraddiag(self, p):
        forces = []
        for i, loc in enumerate(iterable(p.loc)):
            d = self.saved(loc, 'diag_value')
            grad = self.saved(loc, 'diag_grad')
            c = self.kern.gradgrad(d, d).squeeze()
            if self.has_factor:
                self.recalculate(loc)
                f = self.saved(loc, 'diag_fac')
                fg = self.saved(loc, 'diag_facgrad')
                c = (c*(f*f.t()) + fg*fg.t()*self.kern(d, d) +
                     (fg*f.t()*self.kern.rightgrad(d, d)) +
                     (f*fg.t()*self.kern.leftgrad(d, d))).squeeze()
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


def test():
    from theforce.descriptor.atoms import namethem
    from theforce.math.cutoff import PolyCut
    from theforce.regression.kernel import Positive, DotProd, Normed
    from theforce.regression.stationary import RBF
    from theforce.descriptor.atoms import TorchAtoms, AtomsData
    import numpy as np
    torch.set_default_tensor_type(torch.DoubleTensor)

    # create kernel
    kern = PairKernel(RBF(), 18, 10, factor=PolyCut(3.0))
    kerns = [kern]
    namethem(kerns)

    cell = np.ones(3)*10
    positions = np.array([(-1., 0., 0.), (1., 0., 0.),
                          (0., -1., 0.), (0., 1., 0.),
                          (0., 0., -1.1), (0., 0., 1.1),
                          (0., 0., 0.)]) + cell/2

    b = TorchAtoms(positions=positions, numbers=3*[10]+3*[18]+[10], cell=cell,
                   pbc=True, cutoff=3.0, descriptors=kerns)

    # make natoms different in a, b. P.S. add an isolated atom.
    _pos = np.concatenate([positions, [[0., 0., 0.], [3., 5., 5.]]])
    a = TorchAtoms(positions=_pos, numbers=2*[10, 18, 10]+[18, 10, 18], cell=cell,
                   pbc=True, cutoff=3.0, descriptors=kerns)

    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)

    # left/right-grad
    kern([a], [b]).backward()
    test_left = a.xyz.grad.allclose(kern.leftgrad(a, b).view(-1, 3))
    max_left = (a.xyz.grad - kern.leftgrad(a, b).view(-1, 3)).max()
    print("leftgrad: {}  \t max diff: {}".format(test_left, max_left))
    test_right = b.xyz.grad.allclose(kern.rightgrad(a, b).view(-1, 3))
    max_right = (b.xyz.grad - kern.rightgrad(a, b).view(-1, 3)).max()
    print("rightgrad: {} \t max diff: {}".format(test_right, max_right))

    # gradgrad-left
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (kern.leftgrad(a, b).view(-1, 3)*a.xyz).sum().backward()
    v1 = a.xyz.grad.data
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (kern.gradgrad(a, b)*a.xyz.view(-1)[:, None]).sum().backward()
    v2 = a.xyz.grad.data
    print('gradgrad-left: {}'.format(v1.allclose(v2)))

    # gradgrad-right
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (kern.rightgrad(a, b).view(-1, 3)*b.xyz).sum().backward()
    v1 = b.xyz.grad.data
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (kern.gradgrad(a, b)*b.xyz.view(-1)[None]).sum().backward()
    v2 = b.xyz.grad.data
    print('gradgrad-right: {}'.format(v1.allclose(v2)))

    # gradgraddiag
    test_diag = kern.gradgrad(a, a).diag().allclose(kern.gradgraddiag(a))
    print('gradgraddiag: {}'.format(test_diag))


def example():
    from torch import tensor
    from theforce.regression.stationary import RBF
    from theforce.math.cutoff import PolyCut
    from theforce.math.radial import Product, ParamedRepulsiveCore

    factor = Product(PolyCut(1.0), ParamedRepulsiveCore())
    kern = PairKernel(RBF(), 1, 1, factor=factor)
    print(kern.state == eval(kern.state).state)

    d = torch.arange(0.1, 1.5, 0.1).view(-1)
    d.requires_grad = True
    f = eval(factor.state)
    a, b = f(d)
    a.sum().backward()
    print(d.grad.allclose(b))


if __name__ == '__main__':
    test()
    example()

