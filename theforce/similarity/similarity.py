#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import torch
from torch.nn import Module
from torch import zeros, cat
from theforce.util.util import iterable


class SimilarityKernel(Module):

    def __init__(self, kernel):
        super().__init__()
        self.kern = kernel
        self.params = kernel.params

    def forward(self, first, second, operation='func'):
        return cat([cat([getattr(self, operation)(a, b) for a in iterable(first)], dim=0
                        ) for b in iterable(second)], dim=1)


class PairSimilarity(SimilarityKernel):

    def __init__(self, kernel, a, b):
        super().__init__(kernel)
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
                                                  [:, None] * first.dr[m1][..., None]
                                                  ).sum(dim=-1)).view(first.natoms*3, 1) / self.double_count

    def rightgrad(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        return - zeros(second.natoms, 3).index_add(0, second.i[m2],
                                                   (self.kern.rightgrad(first.d[m1], second.d[m2])
                                                    [..., None] * second.dr[m2]).sum(dim=0)
                                                   ).view(1, second.natoms*3) / self.double_count

    def gradgrad(self, first, second):
        m1 = first.select(self.a, self.b, bothways=True)
        m2 = second.select(self.a, self.b, bothways=True)
        return zeros(first.natoms, second.natoms, 3, 3
                     ).index_add(1, second.i[m2], zeros(first.natoms, second.i[m2].size(0), 3, 3
                                                        ).index_add(0, first.i[m1],
                                                                    (first.dr[m1][:, None, :, None]
                                                                     * second.dr[m2][None, :, None, :]
                                                                     * self.kern.gradgrad(
                                                                        first.d[m1], second.d[m2])
                                                                     [..., None, None])
                                                                    )
                                 ).permute(0, 2, 1, 3).contiguous().view(first.natoms*3, second.natoms*3)

