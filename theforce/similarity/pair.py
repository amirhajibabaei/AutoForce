
# coding: utf-8

# In[ ]:


from theforce.similarity.similarity import SimilarityKernel
from torch import zeros, cat, stack
from theforce.util.util import iterable


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

