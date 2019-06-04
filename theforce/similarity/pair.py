
# coding: utf-8

# In[ ]:


from theforce.similarity.similarity import SimilarityKernel
from torch import zeros, cat
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
        c = (p.u[m1][:, None, :, None] * q.u[m2][None, :, None, :]
             * self.kern.gradgrad(p.d[m1], q.d[m2])[..., None, None])
        cc = zeros(p.natoms, q.i[m2].size(0), 3, 3).index_add(0, p.i[m1], c
                                                              ).index_add(0, p.j[m1], -c)
        ccc = zeros(p.natoms, q.natoms, 3, 3).index_add(1, q.i[m2], cc
                                                        ).index_add(1, q.j[m2], -cc)
        return ccc.permute(0, 2, 1, 3).contiguous().view(p.natoms*3, q.natoms*3)


class LogDistanceSimilarity(SimilarityKernel):
    """ Pair energy is assumed as: func(distance). """

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
        c = (p.logd_deriv[m1][:, None, :, None] * q.logd_deriv[m2][None, :, None, :]
             * self.kern.gradgrad(p.logd[m1], q.logd[m2])[..., None, None])
        cc = zeros(p.natoms, q.i[m2].size(0), 3, 3).index_add(0, p.i[m1], c
                                                              ).index_add(0, p.j[m1], -c)
        ccc = zeros(p.natoms, q.natoms, 3, 3).index_add(1, q.i[m2], cc
                                                        ).index_add(1, q.j[m2], -cc)
        return ccc.permute(0, 2, 1, 3).contiguous().view(p.natoms*3, q.natoms*3)


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


# ------------------------------------------------------------------- Deprecated routines
class DistanceSimilarity_deprecated(SimilarityKernel):
    """ Pair energy is assumed as: func(distance). """

    def __init__(self, kernels, a, b):
        super().__init__([kern(dim=1) for kern in iterable(kernels)])
        self.a = a
        self.b = b

    def func(self, p, q):
        m1 = p.select(self.a, self.b, bothways=True)
        m2 = q.select(self.a, self.b, bothways=True)
        return self.kern(p.d[m1], q.d[m2]).sum().view(1, 1) / 4

    def leftgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=True)
        m2 = q.select(self.a, self.b, bothways=True)
        return -zeros(p.natoms, 3).index_add(0, p.i[m1],
                                             (self.kern.leftgrad(p.d[m1], q.d[m2])
                                              [:, None] * p.u[m1][..., None]
                                              ).sum(dim=-1)).view(p.natoms*3, 1) / 2

    def rightgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=True)
        m2 = q.select(self.a, self.b, bothways=True)
        return - zeros(q.natoms, 3).index_add(0, q.i[m2],
                                              (self.kern.rightgrad(p.d[m1], q.d[m2])
                                               [..., None] * q.u[m2]).sum(dim=0)
                                              ).view(1, q.natoms*3) / 2

    def gradgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=True)
        m2 = q.select(self.a, self.b, bothways=True)
        return zeros(p.natoms, q.natoms, 3, 3
                     ).index_add(1, q.i[m2], zeros(p.natoms, q.i[m2].size(0), 3, 3
                                                   ).index_add(0, p.i[m1],
                                                               (p.u[m1][:, None, :, None]
                                                                * q.u[m2][None, :, None, :]
                                                                * self.kern.gradgrad(
                                                                   p.d[m1], q.d[m2])
                                                                [..., None, None])
                                                               )
                                 ).permute(0, 2, 1, 3).contiguous().view(p.natoms*3, q.natoms*3)


class LogDistanceSimilarity_deprecated(SimilarityKernel):
    """ Pair energy is assumed as: func(log-distance). """

    def __init__(self, kernels, a, b):
        super().__init__([kern(dim=1) for kern in iterable(kernels)])
        self.a = a
        self.b = b

    def func(self, p, q):
        m1 = p.select(self.a, self.b, bothways=True)
        m2 = q.select(self.a, self.b, bothways=True)
        return self.kern(p.logd[m1], q.logd[m2]).sum().view(1, 1) / 4

    def leftgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=True)
        m2 = q.select(self.a, self.b, bothways=True)
        return -zeros(p.natoms, 3).index_add(0, p.i[m1],
                                             (self.kern.leftgrad(p.logd[m1], q.logd[m2])
                                              [:, None] * p.logd_deriv[m1][..., None]
                                              ).sum(dim=-1)).view(p.natoms*3, 1) / 2

    def rightgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=True)
        m2 = q.select(self.a, self.b, bothways=True)
        return - zeros(q.natoms, 3).index_add(0, q.i[m2],
                                              (self.kern.rightgrad(p.logd[m1], q.logd[m2])
                                               [..., None] * q.logd_deriv[m2]).sum(dim=0)
                                              ).view(1, q.natoms*3) / 2

    def gradgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=True)
        m2 = q.select(self.a, self.b, bothways=True)
        return zeros(p.natoms, q.natoms, 3, 3
                     ).index_add(1, q.i[m2], zeros(p.natoms, q.i[m2].size(0), 3, 3
                                                   ).index_add(0, p.i[m1],
                                                               (p.logd_deriv[m1][:, None, :, None]
                                                                * q.logd_deriv[m2][None, :, None, :]
                                                                * self.kern.gradgrad(
                                                                   p.logd[m1], q.logd[m2])
                                                                [..., None, None])
                                                               )
                                 ).permute(0, 2, 1, 3).contiguous().view(p.natoms*3, q.natoms*3)


class CoulombPairSimilarity_deprecated(SimilarityKernel):
    """ Pair energy is assumed as: func(distance)/distance. """

    def __init__(self, kernels, a, b):
        super().__init__([kern(dim=1) for kern in iterable(kernels)])
        self.a = a
        self.b = b

    def func(self, p, q):
        m1 = p.select(self.a, self.b, bothways=True)
        m2 = q.select(self.a, self.b, bothways=True)
        k = self.kern(p.d[m1], q.d[m2])
        k = k / (p.d[m1]*q.d[m2].t())
        return k.sum().view(1, 1) / 4

    def leftgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=True)
        m2 = q.select(self.a, self.b, bothways=True)
        lg = self.kern.leftgrad(p.d[m1], q.d[m2])
        lg = lg - (self.kern(p.d[m1], q.d[m2])
                   / p.d[m1])
        lg = lg / (p.d[m1]*q.d[m2].t())
        lg = (lg[:, None] * p.u[m1][..., None]).sum(dim=-1)
        return -zeros(p.natoms, 3).index_add(0, p.i[m1], lg
                                             ).view(p.natoms*3, 1) / 2

    def rightgrad(self, p, q):
        m1 = p.select(self.a, self.b, bothways=True)
        m2 = q.select(self.a, self.b, bothways=True)
        rg = self.kern.rightgrad(p.d[m1], q.d[m2])
        rg = rg - (self.kern(p.d[m1], q.d[m2])
                   / q.d[m2].t())
        rg = rg / (p.d[m1]*q.d[m2].t())
        rg = (rg[..., None] * q.u[m2]).sum(dim=0)
        return - zeros(q.natoms, 3).index_add(0, q.i[m2], rg
                                              ).view(1, q.natoms*3) / 2

    def gradgrad(self, p, q):
        raise NotImplementedError(
            'PairSimilarity: gradgrad is not implemented yet!')

