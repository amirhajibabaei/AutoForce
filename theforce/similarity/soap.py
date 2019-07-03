
# coding: utf-8

# In[ ]:


from theforce.similarity.similarity import SimilarityKernel
from theforce.math.soap import SeriesSoap
from theforce.util.util import iterable
import torch


class SoapKernel(SimilarityKernel):
    """
    Notes:
    1. the descriptor vectors for all the species will be concatenated, 
    and then will be normalized i.e. "normalize=False" when constructing
    the descriptor for a single species.
    2. "scale" and "actual" keywords in the SeriesSoap control the resolution.
    3. Each species can have its own SeriesSoap object.
    """

    def __init__(self, kernel, a, b, lmax, nmax, radial):
        super().__init__(kernel)
        self.a = a
        self.b = sorted(iterable(b))
        self.descriptor = SeriesSoap(lmax, nmax, radial, scale=None, actual=False,
                                     normalize=False, cutcorners=0, symm=False)
        self.soapdim = self.descriptor.mask.sum()*(lmax+1)
        self.dim = len(self.b)*self.soapdim

    @property
    def state_args(self):
        return super().state_args + ', {}, {}, {}, {}, {}'.format(
            self.a, self.b, self.descriptor.abs.ylm.lmax, self.descriptor.abs.nmax,
            self.descriptor.abs.radial.state)

    def precalculate(self, loc):
        idx = torch.arange(loc._j.size(0)).long()
        zero = torch.zeros(self.soapdim, loc._j.size(0), 3)
        dat = []
        for b in self.b:
            loc.select(self.a, b, bothways=True, in_place=True)
            d, _grad = self.descriptor(loc.r)
            grad = zero.index_add(1, idx[loc._m], _grad)
            dat += [(d, grad)]
        d, grad = (torch.cat(a) for a in zip(*dat))
        # normalize
        norm = d.norm()
        if norm > 0.0:
            d = d/norm
            grad = grad/norm
            grad = (grad - d[..., None, None] *
                    (d[..., None, None] * grad).sum(dim=0))
        grad = torch.cat([grad, -grad.sum(dim=1, keepdim=True)], dim=1)
        j = torch.cat([loc._j, loc._i.unique()])
        # save
        data = {'value': d[None], 'grad': grad, 'j': j}
        self.save_for_later(loc, data)

    def func(self, p, q):
        d = self.saved(p, 'value')
        dd = self.saved(q, 'value')
        c = self.kern(d, dd)
        return c.sum().view(1, 1)

    def leftgrad(self, p, q):
        d = self.saved(p, 'value')
        dd = self.saved(q, 'value')
        c = self.kern.leftgrad(d, dd)
        g = torch.zeros(p.natoms, 3)
        for i, loc in enumerate(p):
            grad = self.saved(loc, 'grad')
            j = self.saved(loc, 'j')
            t = (c[:, i][..., None, None]*grad[:, None]).sum(dim=(0, 1))
            g = g.index_add(0, j, t)
        return g.view(-1, 1)

    def rightgrad(self, p, q):
        d = self.saved(p, 'value')
        dd = self.saved(q, 'value')
        c = self.kern.rightgrad(d, dd)
        g = torch.zeros(p.natoms, 3)
        for i, loc in enumerate(q):
            grad = self.saved(loc, 'grad')
            j = self.saved(loc, 'j')
            t = (c[..., i][..., None, None]*grad[:, None]).sum(dim=(0, 1))
            g = g.index_add(0, j, t)
        return g.view(1, -1)

    def gradgrad(self, p, q):
        raise NotImplementedError('Not defined yet')

    def gradgraddiag(self, p):
        raise NotImplementedError('Not defined yet')


def test_grad():
    from theforce.descriptor.atoms import namethem
    from theforce.math.cutoff import PolyCut
    from theforce.regression.kernel import Positive, DotProd
    from theforce.descriptor.atoms import TorchAtoms, AtomsData
    import numpy as np
    torch.set_default_tensor_type(torch.DoubleTensor)

    # create kernel
    kern = Positive(1.0) * (DotProd()+Positive(0.01))**4
    soap = SoapKernel(kern, 10, (18, 10), 2, 2, PolyCut(3.0))
    namethem([soap])

    # create to atomic systems
    # Note that when one of the displacement vectors becomes is exactly along the z-axis
    # because of singularity some inconsistensies exist with autograd.
    # For this reason we add a small random number to positions, until that bug is fixed.
    cell = np.ones(3)*10
    positions = np.array([(-1., 0., 0.), (1., 0., 0.),
                          (0., -1., 0.), (0., 1., 0.),
                          (0., 0., -1.), (0., 0., 1.),
                          (0., 0., 0.)]) + cell/2 + np.random.uniform(-0.1, 0.1, size=(7, 3))

    b = TorchAtoms(positions=positions, numbers=3*[10]+3*[18]+[10], cell=cell,
                   pbc=True, cutoff=3.0, descriptors=[soap])
    a = TorchAtoms(positions=positions, numbers=2*[10, 18, 10]+[18], cell=cell,
                   pbc=True, cutoff=3.0, descriptors=[soap])
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    soap([a], [b]).backward()

    test_left = a.xyz.grad.allclose(soap.leftgrad(a, b).view(-1, 3))
    max_left = (a.xyz.grad - soap.leftgrad(a, b).view(-1, 3)).max()
    print("leftgrad: {}  \t max diff: {}".format(test_left, max_left))
    test_right = b.xyz.grad.allclose(soap.rightgrad(a, b).view(-1, 3))
    max_right = (b.xyz.grad - soap.rightgrad(a, b).view(-1, 3)).max()
    print("rightgrad: {} \t max diff: {}".format(test_right, max_right))


def example():
    from theforce.regression.kernel import Positive, DotProd
    from theforce.math.cutoff import PolyCut
    kern = (Positive(1.0, requires_grad=True) *
            (DotProd() + Positive(0.01, requires_grad=True))**4)
    soap = SoapKernel(kern, 10, (18, 10), 2, 2, PolyCut(3.0))


if __name__ == '__main__':
    example()
    test_grad()

