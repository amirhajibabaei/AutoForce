
# coding: utf-8

# In[ ]:


#import torch
from torch.nn import Module
from torch import zeros, cat
from theforce.util.util import iterable
from theforce.regression.gp import Covariance


class SimilarityKernel(Module):

    def __init__(self, kernels):
        super().__init__()
        self.kern = Covariance(kernels)
        self.params = self.kern.params

    def forward(self, first, second, operation='func'):
        return cat([cat([getattr(self, operation)(a, b) for a in iterable(first)], dim=0
                        ) for b in iterable(second)], dim=1)


class DSimilarity(SimilarityKernel):

    def __init__(self, kernels, a, b):
        super().__init__([kern(dim=1) for kern in iterable(kernels)])
        self.a = a
        self.b = b
        self.double_count = 2

    def func(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        return self.kern(first.d[m1], second.d[m2]).sum().view(1, 1) / self.double_count**2

    def leftgrad(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        return -zeros(first.natoms, 3).index_add(0, first.i[m1],
                                                 (self.kern.leftgrad(first.d[m1], second.d[m2])
                                                  [:, None] * first.u[m1][..., None]
                                                  ).sum(dim=-1)).view(first.natoms*3, 1) / self.double_count

    def rightgrad(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        return - zeros(second.natoms, 3).index_add(0, second.i[m2],
                                                   (self.kern.rightgrad(first.d[m1], second.d[m2])
                                                    [..., None] * second.u[m2]).sum(dim=0)
                                                   ).view(1, second.natoms*3) / self.double_count

    def gradgrad(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        return zeros(first.natoms, second.natoms, 3, 3
                     ).index_add(1, second.i[m2], zeros(first.natoms, second.i[m2].size(0), 3, 3
                                                        ).index_add(0, first.i[m1],
                                                                    (first.u[m1][:, None, :, None]
                                                                     * second.u[m2][None, :, None, :]
                                                                     * self.kern.gradgrad(
                                                                        first.d[m1], second.d[m2])
                                                                     [..., None, None])
                                                                    )
                                 ).permute(0, 2, 1, 3).contiguous().view(first.natoms*3, second.natoms*3)


class LogDSimilarity(SimilarityKernel):

    def __init__(self, kernels, a, b):
        super().__init__([kern(dim=1) for kern in iterable(kernels)])
        self.a = a
        self.b = b
        self.double_count = 2

    def func(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        return self.kern(first.logd[m1], second.logd[m2]).sum().view(1, 1) / self.double_count**2

    def leftgrad(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        return -zeros(first.natoms, 3).index_add(0, first.i[m1],
                                                 (self.kern.leftgrad(first.logd[m1], second.logd[m2])
                                                  [:, None] * first.logd_deriv[m1][..., None]
                                                  ).sum(dim=-1)).view(first.natoms*3, 1) / self.double_count

    def rightgrad(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        return - zeros(second.natoms, 3).index_add(0, second.i[m2],
                                                   (self.kern.rightgrad(first.logd[m1], second.logd[m2])
                                                    [..., None] * second.logd_deriv[m2]).sum(dim=0)
                                                   ).view(1, second.natoms*3) / self.double_count

    def gradgrad(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        return zeros(first.natoms, second.natoms, 3, 3
                     ).index_add(1, second.i[m2], zeros(first.natoms, second.i[m2].size(0), 3, 3
                                                        ).index_add(0, first.i[m1],
                                                                    (first.logd_deriv[m1][:, None, :, None]
                                                                     * second.logd_deriv[m2][None, :, None, :]
                                                                     * self.kern.gradgrad(
                                                                        first.logd[m1], second.logd[m2])
                                                                     [..., None, None])
                                                                    )
                                 ).permute(0, 2, 1, 3).contiguous().view(first.natoms*3, second.natoms*3)


class PairSimilarity(SimilarityKernel):

    def __init__(self, kernels, a, b):
        super().__init__([kern(dim=1) for kern in iterable(kernels)])
        self.a = a
        self.b = b

    def func(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        k = self.kern(first.d[m1], second.d[m2])
        k = k / (first.d[m1]*second.d[m2].t())
        return k.sum().view(1, 1) / 4

    def leftgrad(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        lg = self.kern.leftgrad(first.d[m1], second.d[m2])
        lg = lg - (self.kern(first.d[m1], second.d[m2])
                   / first.d[m1])
        lg = lg / (first.d[m1]*second.d[m2].t())
        lg = (lg[:, None] * first.u[m1][..., None]).sum(dim=-1)
        return -zeros(first.natoms, 3).index_add(0, first.i[m1], lg
                                                 ).view(first.natoms*3, 1) / 2

    def rightgrad(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        rg = self.kern.rightgrad(first.d[m1], second.d[m2])
        rg = rg - (self.kern(first.d[m1], second.d[m2])
                   / second.d[m2].t())
        rg = rg / (first.d[m1]*second.d[m2].t())
        rg = (rg[..., None] * second.u[m2]).sum(dim=0)
        return - zeros(second.natoms, 3).index_add(0, second.i[m2], rg
                                                   ).view(1, second.natoms*3) / 2

    def gradgrad(self, first, second):
        raise NotImplementedError(
            'PairSimilarity: gradgrad is not implemented yet!')

