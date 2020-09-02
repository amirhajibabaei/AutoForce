# +
from theforce.similarity.similarity import SimilarityKernel
from theforce.regression.kernel import Kernel
from theforce.math.cutoff import PolyCut
from theforce.math.soap import UniversalSoap
from theforce.util.util import iterable
import torch


class Chemical(Kernel):

    def __init__(self):
        super().__init__()
        self.params = []


class DiracDeltaChemical(Chemical):

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        if a == b:
            return 1.
        else:
            return 0.

    @property
    def state_args(self):
        return ''


def indices(a, b):
    i = a._indices()
    j = b._indices()
    size = torch.stack([i.max(dim=1).values, j.max(dim=1).values]).max(dim=0)
    size = [s+1 for s in size.values.tolist()]
    k = (torch.sparse_coo_tensor(i, torch.ones(i.size(1)), size=size) *
         torch.sparse_coo_tensor(j, torch.ones(j.size(1)), size=size))._indices()
    return k


class EqAll:
    def __init__(self, exceptions=[]):
        self.exceptions = exceptions

    def __eq__(self, val):
        return val not in self.exceptions


class UniversalSoapKernel(SimilarityKernel):

    def __init__(self, lmax, nmax, exponent, cutoff, atomic_unit=None, chemical=None, normalize=True,
                 a=None, a_not=[]):
        if chemical is None:
            chemical = DiracDeltaChemical()
        super().__init__(chemical)
        radial = PolyCut(cutoff) if type(cutoff) == float else cutoff
        self.descriptor = UniversalSoap(
            lmax, nmax, radial, atomic_unit=atomic_unit, normalize=normalize)
        self.exponent = exponent
        self.dim = self.descriptor.dim
        self._args = '{}, {}, {}, {}, atomic_unit={}, chemical={}, normalize={}, a={}, a_not={}'.format(
            lmax, nmax, exponent, radial.state, atomic_unit, self.kern.state, normalize, a, a_not)
        self._a = EqAll(a_not) if a is None else a

        self.cutoff = radial.rc

    @property
    def a(self):
        return self._a

    @property
    def state_args(self):
        return self._args

    def call_descriptor(self, loc, grad):
        return self.descriptor(loc._r, loc._b, grad=grad)

    def precalculate(self, loc, dont_save_grads=True):
        if loc._r.size(0) > 0 and loc.number == self.a:
            d = self.call_descriptor(loc, grad=not dont_save_grads)
            self.save_for_later(loc, {'value': d} if dont_save_grads else
                                {'value': d[0],
                                 'grad': d[1]})
        else:
            self.save_for_later(loc, {'value': None, 'grad': None})

    def get_func(self, _p, _q):
        c = torch.tensor(0.)
        for p in iterable(_p):
            d = self.saved(p, 'value')
            if d is None:
                continue
            for q in iterable(_q):
                alch = self.kern(p.number, q.number)
                if alch > 0.:
                    dd = self.saved(q, 'value')
                    if dd is None:
                        continue
                    c += alch * torch.sparse.sum(d*dd)**self.exponent
        return c.view(1, 1)

    def get_leftgrad(self, _p, _q):
        g = torch.zeros(_p.natoms, 3)
        for p in iterable(_p):
            d = self.saved(p, 'value')
            if d is None:
                continue
            grad = self.saved(p, 'grad')
            i = p.index
            j = p._j
            for q in iterable(_q):
                alch = self.kern(p.number, q.number)
                if alch > 0.:
                    dd = self.saved(q, 'value')
                    if dd is None:
                        continue
                    dg = torch.stack([(dd[i, j][:, None, None]*grad[i, j]).sum(dim=0)
                                      for i, j in zip(*indices(dd, grad))]).sum(dim=0)
                    c = torch.sparse.sum(d*dd)
                    f = alch * self.exponent*c**(self.exponent-1)*dg
                    g[i] -= f.sum(dim=0)
                    g[j] += f
        return g.view(-1, 1)

    def get_rightgrad(self, p, q):
        return self.get_leftgrad(q, p).t()


def test_grad():
    from theforce.descriptor.atoms import namethem
    from theforce.descriptor.atoms import TorchAtoms, AtomsData
    import numpy as np

    soap = UniversalSoapKernel(2, 2, 4, 3.)
    soap = eval(soap.state)
    namethem([soap])

    #
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
    if 1:
        soap([a], [b]).backward()
        test_left = a.xyz.grad.allclose(soap.leftgrad(a, b).view(-1, 3))
        max_left = (a.xyz.grad - soap.leftgrad(a, b).view(-1, 3)).max()
        print("leftgrad: {}  \t max diff: {}".format(test_left, max_left))
        test_right = b.xyz.grad.allclose(soap.rightgrad(a, b).view(-1, 3))
        max_right = (b.xyz.grad - soap.rightgrad(a, b).view(-1, 3)).max()
        print("rightgrad: {} \t max diff: {}".format(test_right, max_right))

    #
    if 1:
        data = AtomsData([a, b])
        inducing = data.sample_locals(5)
        inducing.stage(descriptors=[soap])
        soap(data, data)
        soap(data, inducing)
        soap(inducing, inducing)


if __name__ == '__main__':
    test_grad()
