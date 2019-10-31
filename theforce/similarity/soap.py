#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from theforce.similarity.similarity import SimilarityKernel
from theforce.math.soap import RealSeriesSoap, TailoredSoap, NormalizedSoap, MultiSoap
from theforce.util.util import iterable
import torch


class SoapKernel(SimilarityKernel):

    def __init__(self, kernel, a, b, lmax, nmax, radial, atomic_unit=None):
        super().__init__(kernel)
        self.a = a
        self.b = sorted(iterable(b))
        if atomic_unit == None or type(atomic_unit) == float or type(atomic_unit) == int:
            units = {_b: atomic_unit for _b in self.b}
        elif type(atomic_unit) == list or type(atomic_unit) == tuple:
            # if au is a list or a tuple, so should b! self.b is sorted so we shoud use (arg) b.
            units = {_b: au for _b, au in zip(*[b, atomic_unit])}
        elif type(atomic_unit) == dict:
            units = {_b: atomic_unit[(a, _b)] if (a, _b) in atomic_unit else atomic_unit[(_b, a)]
                     for _b in self.b}
        self.descriptor = MultiSoap([TailoredSoap(RealSeriesSoap(
            lmax, nmax, radial, atomic_unit=units[_b])) for _b in self.b])
        self.dim = self.descriptor.dim
        self._args = '{}, {}, {}, {}, {}, atomic_unit={}'.format(
            a, b, lmax, nmax, radial.state, atomic_unit)

    @property
    def state_args(self):
        return super().state_args + ', ' + self._args

    def precalculate(self, loc, dont_save_grads=False):
        if (self.a == loc._a.unique()).all():
            masks = [loc.select(self.a, b, bothways=True) for b in self.b]
            d, grad = self.descriptor(loc._r, masks, grad=True)
            d = d[None]
            grad = torch.cat([grad, -grad.sum(dim=1, keepdim=True)], dim=1)
            j = torch.cat([loc._j, loc._i.unique()])
            if j.numel() > 0:
                empty = torch.tensor([False])
            else:
                empty = torch.tensor([True])
                del d, grad
                d = torch.zeros(0, self.dim)
                grad = torch.zeros(self.dim, 0, 3)
        else:
            empty = torch.tensor([True])
            d = torch.zeros(0, self.dim)
            grad = torch.zeros(self.dim, 0, 3)
            j = torch.empty(0).long()
        # save
        if dont_save_grads:
            del grad, j, empty
            data = {'value': d}
        else:
            data = {'value': d, 'grad': grad, 'j': j, 'empty': empty}
        self.save_for_later(loc, data)

    def get_func(self, p, q):
        d = self.saved(p, 'value')
        dd = self.saved(q, 'value')
        c = self.kern(d, dd)
        return c.sum().view(1, 1)

    def get_leftgrad(self, p, q):
        d = self.saved(p, 'value')
        dd = self.saved(q, 'value')
        empty = self.saved(p, 'empty')
        c = self.kern.leftgrad(d, dd)
        g = torch.zeros(p.natoms, 3)
        _i = 0
        for i, loc in enumerate(p):
            if not empty[i]:
                grad = self.saved(loc, 'grad')
                j = self.saved(loc, 'j')
                t = (c[:, _i][..., None, None]*grad[:, None]).sum(dim=(0, 1))
                g = g.index_add(0, j, t)
                _i += 1
        return g.view(-1, 1)

    def get_rightgrad(self, p, q):
        d = self.saved(p, 'value')
        dd = self.saved(q, 'value')
        empty = self.saved(q, 'empty')
        c = self.kern.rightgrad(d, dd)
        g = torch.zeros(q.natoms, 3)
        _i = 0
        for i, loc in enumerate(q):
            if not empty[i]:
                grad = self.saved(loc, 'grad')
                j = self.saved(loc, 'j')
                t = (c[..., _i][..., None, None]*grad[:, None]).sum(dim=(0, 1))
                g = g.index_add(0, j, t)
                _i += 1
        return g.view(1, -1)

    def graddata(self, p):
        empty = self.saved(p, 'empty')
        i = []
        j = []
        grad = []
        _i = 0
        for k, loc in enumerate(p):
            if not empty[k]:
                grad += [self.saved(loc, 'grad')]
                _j = self.saved(loc, 'j')
                j += [_j]
                i += [torch.full_like(_j, _i)]
                _i += 1
        i = torch.cat(i)
        j = torch.cat(j)
        grad = torch.cat(grad, dim=1)
        return i, j, grad

    def get_gradgrad(self, p, q):
        d1 = self.saved(p, 'value')
        d2 = self.saved(q, 'value')
        c = self.kern.gradgrad(d1, d2)
        i1, j1, grad1 = self.graddata(p)
        i2, j2, grad2 = self.graddata(q)
        a = ((grad1[:, None, :, None, :, None]*grad2[None, :, None, :, None]))
        b = c.index_select(2, i1).index_select(3, i2)[..., None, None]
        g = (a*b).sum(dim=(0, 1)).permute(0, 2, 1, 3)
        f = torch.zeros(p.natoms, 3, j2.size(0), 3).index_add(0, j1, g)
        h = torch.zeros(p.natoms, 3, q.natoms, 3).index_add(2, j2, f)
        return h.view(p.natoms*3, q.natoms*3)

    def get_gradgraddiag(self, p):
        d = self.saved(p, 'value')
        empty = self.saved(p, 'empty')
        c = self.kern.gradgrad(d, d)
        i, j, grad = self.graddata(p)
        h = torch.zeros(p.natoms, 3)
        for jj in torch.unique(j):
            m = j == jj
            ii = i[m]
            a = grad[:, m][:, None, :, None]*grad[:, m][None, :, None]
            b = c.index_select(2, ii).index_select(3, ii).unsqueeze(-1)
            f = (a*b).sum(dim=(0, 1, 2, 3)).view(1, 3)
            h[jj] = f
        return h.view(-1)


class NormedSoapKernel(SoapKernel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.descriptor = NormalizedSoap(self.descriptor)


def test_grad():
    from theforce.descriptor.atoms import namethem
    from theforce.math.cutoff import PolyCut
    from theforce.regression.kernel import Positive, DotProd, Normed
    from theforce.regression.stationary import RBF
    from theforce.descriptor.atoms import TorchAtoms, AtomsData
    import numpy as np
    torch.set_default_tensor_type(torch.DoubleTensor)

    # create kernel
    kern = Positive(1.0) * Normed(DotProd())**4
    #kern = RBF()
    #soap = SoapKernel(kern, 10, (18, 10), 2, 2, PolyCut(3.0))
    soap = NormedSoapKernel(DotProd()**4, 10, (18, 10), 2, 2, PolyCut(3.0))
    namethem([soap])

    # create atomic systems
    # Note that when one of the displacement vectors becomes is exactly along the z-axis
    # because of singularity some inconsistensies exist with autograd.
    # For this reason we add a small random number to positions, until that bug is fixed.
    cell = np.ones(3)*10
    positions = np.array([(-1., 0., 0.), (1., 0., 0.),
                          (0., -1., 0.), (0., 1., 0.),
                          (0., 0., -1.), (0., 0., 1.0),
                          (0., 0., 0.)]) + cell/2 + np.random.uniform(-0.1, 0.1, size=(7, 3))

    b = TorchAtoms(positions=positions, numbers=3*[10]+3*[18]+[10], cell=cell,
                   pbc=True, cutoff=3.0, descriptors=[soap])
    # make natoms different in a, b. P.S. add an isolated atom.
    _pos = np.concatenate([positions, [[0., 0., 0.], [3., 5., 5.]]])
    a = TorchAtoms(positions=_pos, numbers=2*[10, 18, 10]+[18, 10, 18], cell=cell,
                   pbc=True, cutoff=3.0, descriptors=[soap])

    # left/right-grad
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    soap([a], [b]).backward()
    test_left = a.xyz.grad.allclose(soap.leftgrad(a, b).view(-1, 3))
    max_left = (a.xyz.grad - soap.leftgrad(a, b).view(-1, 3)).max()
    print("leftgrad: {}  \t max diff: {}".format(test_left, max_left))
    test_right = b.xyz.grad.allclose(soap.rightgrad(a, b).view(-1, 3))
    max_right = (b.xyz.grad - soap.rightgrad(a, b).view(-1, 3)).max()
    print("rightgrad: {} \t max diff: {}".format(test_right, max_right))

    # gradgrad-left
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (soap.leftgrad(a, b).view(-1, 3)*a.xyz).sum().backward()
    v1 = a.xyz.grad.data
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (soap.gradgrad(a, b)*a.xyz.view(-1)[:, None]).sum().backward()
    v2 = a.xyz.grad.data
    print('gradgrad-left: {}'.format(v1.allclose(v2)))

    # gradgrad-right
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (soap.rightgrad(a, b).view(-1, 3)*b.xyz).sum().backward()
    v1 = b.xyz.grad.data
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (soap.gradgrad(a, b)*b.xyz.view(-1)[None]).sum().backward()
    v2 = b.xyz.grad.data
    print('gradgradright: {}'.format(v1.allclose(v2)))

    # gradgraddiag
    test_diag = soap.gradgrad(a, a).diag().allclose(soap.gradgraddiag(a))
    print('gradgraddiag: {}'.format(test_diag))


def example():
    from theforce.regression.kernel import Positive, DotProd, Mul, Add, Pow
    from theforce.math.cutoff import PolyCut
    kern = (Positive(1.0, requires_grad=True) *
            (DotProd() + Positive(0.01, requires_grad=True))**0.1)
    soap = SoapKernel(kern, 10, (18, 10), 2, 2, PolyCut(3.0))
    assert eval(soap.state).state == soap.state


if __name__ == '__main__':
    example()
    test_grad()

