
# coding: utf-8

# In[ ]:


from theforce.similarity.similarity import SimilarityKernel
from torch import zeros, cat, stack
from theforce.util.util import iterable


# ---------------------------------------------------------------- defined for TorchAtoms
class PairSimilarityKernel(SimilarityKernel):

    def __init__(self, kernels, a, b):
        super().__init__([kern(dim=1) for kern in iterable(kernels)])
        self.a = a
        self.b = b

    def descriptor(self, r):
        raise NotImplementedError(
            'descriptor shoud be implemented in a child class')

    def save_for_later(self, loc, keyvals):
        for key, val in keyvals.items():
            setattr(loc, self.name+'_'+key, val)

    def saved(self, atoms_or_loc, key):
        return getattr(atoms_or_loc, self.name+'_'+key)

    def calculate(self, loc):
        loc.select(self.a, self.b, bothways=True)
        d, grad = self.descriptor(loc.r)
        data = {'diag_value': d, 'diag_grad': grad}
        self.save_for_later(loc, data)
        m = loc.j > loc.i
        data = {'value': d[m], 'grad': grad[m], 'i': loc.i[m], 'j': loc.j[m]}
        self.save_for_later(loc, data)
        if hasattr(self, 'factor'):
            fac, facgrad = self.factor(d)
            self.save_for_later(loc, {'diag_fac': fac, 'diag_facgrad': facgrad,
                                      'fac': fac[m], 'facgrad': facgrad[m]})
            self.has_factor = True
        else:
            self.has_factor = False

    def func(self, p, q):
        d = self.saved(p, 'value')
        dd = self.saved(q, 'value')
        c = self.kern(d, dd)
        if self.has_factor:
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
    def __init__(self, kernels, a, b):
        super().__init__(kernels, a, b)

    def descriptor(self, r):
        d = (r**2).sum(dim=-1).sqrt().view(-1, 1)
        grad = r/d
        return d, grad


class LogDistanceKernel(PairSimilarityKernel):
    def __init__(self, kernels, a, b):
        super().__init__(kernels, a, b)

    def descriptor(self, r):
        d = (r**2).sum(dim=-1).sqrt().view(-1, 1)
        grad = r/d**2
        return d.log(), grad


class RepulsiveCoreKernel(DistanceKernel):
    def __init__(self, kernels, a, b, eta=1):
        super().__init__(kernels, a, b)
        self.eta = eta

    def factor(self, d):
        return 1./d**self.eta, -self.eta/d**(self.eta+1)


# -------------------------------------------------------------- defined for Systems
class DistanceSimilarity(SimilarityKernel):
    """ Pair energy is assumed as: func(distance). """

    def __init__(self, kernels, a, b):
        super().__init__([kern(dim=1) for kern in iterable(kernels)])
        self.a = a
        self.b = b

    def func(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        return self.kern(p.d[m1], q.d[m2]).sum().view(1, 1)

    def leftgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = (self.kern.leftgrad(p.d[m1], q.d[m2])[:, None] *
             p.u[m1][..., None]).sum(dim=-1)
        return zeros(p.natoms, 3).index_add(0, p.i[m1], -c).index_add(0, p.j[m1], c).view(-1, 1)

    def rightgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = (self.kern.rightgrad(p.d[m1], q.d[m2])[..., None] *
             q.u[m2]).sum(dim=0)
        return zeros(q.natoms, 3).index_add(0, q.i[m2], -c).index_add(0, q.j[m2], c).view(1, -1)

    def gradgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = (p.u[m1][..., None, None] * q.u[m2] *
             self.kern.gradgrad(p.d[m1], q.d[m2])[:, None, :, None])
        cc = zeros(p.natoms, 3, q.i[m2].size(0), 3).index_add(0, p.i[m1], c
                                                              ).index_add(0, p.j[m1], -c)
        ccc = zeros(p.natoms, 3, q.natoms, 3).index_add(2, q.i[m2], cc
                                                        ).index_add(2, q.j[m2], -cc)
        return ccc.view(p.natoms*3, q.natoms*3)

    def gradgraddiag(self, p):
        m1 = p.select(self.a, self.b, bothways=True)
        i, counts = p.i[m1].unique(return_counts=True)
        _d = p.d[m1].split_with_sizes(counts.tolist())
        _u = p.u[m1].split_with_sizes(counts.tolist())
        c = stack([(self.kern.gradgrad(d, d)[..., None] * u[None, ] * u[:, None]).sum(dim=(0, 1))
                   for d, u in zip(*[_d, _u])])
        return zeros(p.natoms, 3).index_add(0, i, c).view(-1)


class LogDistanceSimilarity(SimilarityKernel):
    """ Pair energy is assumed as: func(log-distance). """

    def __init__(self, kernels, a, b):
        super().__init__([kern(dim=1) for kern in iterable(kernels)])
        self.a = a
        self.b = b

    def func(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        return self.kern(p.logd[m1], q.logd[m2]).sum().view(1, 1)

    def leftgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = (self.kern.leftgrad(p.logd[m1], q.logd[m2])[:, None] *
             p.logd_deriv[m1][..., None]).sum(dim=-1)
        return zeros(p.natoms, 3).index_add(0, p.i[m1], -c).index_add(0, p.j[m1], c).view(-1, 1)

    def rightgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = (self.kern.rightgrad(p.logd[m1], q.logd[m2])[..., None] *
             q.logd_deriv[m2]).sum(dim=0)
        return zeros(q.natoms, 3).index_add(0, q.i[m2], -c).index_add(0, q.j[m2], c).view(1, -1)

    def gradgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = (p.logd_deriv[m1][..., None, None] * q.logd_deriv[m2] *
             self.kern.gradgrad(p.logd[m1], q.logd[m2])[:, None, :, None])
        cc = zeros(p.natoms, 3, q.i[m2].size(0), 3).index_add(0, p.i[m1], c
                                                              ).index_add(0, p.j[m1], -c)
        ccc = zeros(p.natoms, 3, q.natoms, 3).index_add(2, q.i[m2], cc
                                                        ).index_add(2, q.j[m2], -cc)
        return ccc.view(p.natoms*3, q.natoms*3)

    def gradgraddiag(self, p):
        m1 = p.select(self.a, self.b, bothways=True)
        i, counts = p.i[m1].unique(return_counts=True)
        _d = p.logd[m1].split_with_sizes(counts.tolist())
        _u = p.logd_deriv[m1].split_with_sizes(counts.tolist())
        c = stack([(self.kern.gradgrad(d, d)[..., None] * u[None, ] * u[:, None]).sum(dim=(0, 1))
                   for d, u in zip(*[_d, _u])])
        return zeros(p.natoms, 3).index_add(0, i, c).view(-1)


class CoulombPairSimilarity(SimilarityKernel):
    """ Pair energy is assumed as: func(distance)/distance. """

    def __init__(self, kernels, a, b):
        super().__init__([kern(dim=1) for kern in iterable(kernels)])
        self.a = a
        self.b = b

    def func(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = self.kern(p.d[m1], q.d[m2])
        c = c / (p.d[m1]*q.d[m2].t())
        return c.sum().view(1, 1)

    def leftgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = (self.kern.leftgrad(p.d[m1], q.d[m2]) -
             self.kern(p.d[m1], q.d[m2])/p.d[m1])/(p.d[m1]*q.d[m2].t())
        c = (c[:, None] * p.u[m1][..., None]).sum(dim=-1)
        return -zeros(p.natoms, 3).index_add(0, p.i[m1], c).index_add(0, p.j[m1], -c).view(-1, 1)

    def rightgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = (self.kern.rightgrad(p.d[m1], q.d[m2]) -
             self.kern(p.d[m1], q.d[m2])/q.d[m2].t())/(p.d[m1]*q.d[m2].t())
        c = (c[..., None] * q.u[m2]).sum(dim=0)
        return -zeros(q.natoms, 3).index_add(0, q.i[m2], c).index_add(0, q.j[m2], -c).view(1, -1)

    def gradgrad(self, p, q):
        raise NotImplementedError(
            'PairSimilarity: gradgrad is not implemented yet!')

    def gradgraddiag(self, p):
        m1 = p.select(self.a, self.b, bothways=True)
        i, counts = p.i[m1].unique(return_counts=True)
        _d = p.d[m1].split_with_sizes(counts.tolist())
        _u = p.u[m1].split_with_sizes(counts.tolist())
        c = []
        for d, u in zip(*[_d, _u]):
            ddt = d*d.t()
            c += [((
                (self.kern(d, d)/ddt
                 - self.kern.rightgrad(d, d)/d
                 - self.kern.leftgrad(d, d)/d.t()
                 + self.kern.gradgrad(d, d))/ddt
            )[..., None] * u[None, ] * u[:, None]).sum(dim=(0, 1))]
        c = stack(c)
        return zeros(p.natoms, 3).index_add(0, i, c).view(-1)


class RepulsiveCoreSimilarity(SimilarityKernel):
    """ Pair energy is assumed as: func(distance)/distance^eta. """

    def __init__(self, kernels, a, b, eta=1):
        super().__init__([kern(dim=1) for kern in iterable(kernels)])
        self.a = a
        self.b = b
        self.eta = eta

    def func(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = self.kern(p.d[m1], q.d[m2])
        c = c / (p.d[m1]*q.d[m2].t())**self.eta
        return c.sum().view(1, 1)

    def leftgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = (self.kern.leftgrad(p.d[m1], q.d[m2]) -
             self.eta*self.kern(p.d[m1], q.d[m2])/p.d[m1])/(p.d[m1]*q.d[m2].t())**self.eta
        c = (c[:, None] * p.u[m1][..., None]).sum(dim=-1)
        return -zeros(p.natoms, 3).index_add(0, p.i[m1], c).index_add(0, p.j[m1], -c).view(-1, 1)

    def rightgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=False)
        m2 = q.select(self.a, self.b, bothways=False)
        c = (self.kern.rightgrad(p.d[m1], q.d[m2]) -
             self.eta*self.kern(p.d[m1], q.d[m2])/q.d[m2].t())/(p.d[m1]*q.d[m2].t())**self.eta
        c = (c[..., None] * q.u[m2]).sum(dim=0)
        return -zeros(q.natoms, 3).index_add(0, q.i[m2], c).index_add(0, q.j[m2], -c).view(1, -1)

    def gradgrad(self, p, q):
        raise NotImplementedError(
            'PairSimilarity: gradgrad is not implemented yet!')

    def gradgraddiag(self, p):
        m1 = p.select(self.a, self.b, bothways=True)
        i, counts = p.i[m1].unique(return_counts=True)
        _d = p.d[m1].split_with_sizes(counts.tolist())
        _u = p.u[m1].split_with_sizes(counts.tolist())
        c = []
        for d, u in zip(*[_d, _u]):
            ddt = d*d.t()
            c += [((
                (self.eta**2*self.kern(d, d)/ddt
                 - self.eta*self.kern.rightgrad(d, d)/d
                 - self.eta*self.kern.leftgrad(d, d)/d.t()
                 + self.kern.gradgrad(d, d))/ddt**self.eta
            )[..., None] * u[None, ] * u[:, None]).sum(dim=(0, 1))]
        c = stack(c)
        return zeros(p.natoms, 3).index_add(0, i, c).view(-1)

