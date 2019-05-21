
# coding: utf-8

# In[ ]:


from torch.nn import Module


def iterable(a):
    if hasattr(a, '__iter__'):
        return a
    else:
        return (a, )


class SimilarityKernel(Module):

    def __init__(self, kernel):
        super().__init__()
        self.kern = kernel
        self.params = kernel.params

    def forward(self, first, second, operation='func'):
        return torch.cat([torch.cat([getattr(self, operation)(a, b) for a in iterable(first)], dim=0
                                    ) for b in iterable(second)], dim=1)


class PairSimilarity(SimilarityKernel):

    def __init__(self, kernel):
        super().__init__(kernel)

    def func(self, first, second):
        return 0.25 * self.kern(first.d, second.d).sum().view(1, 1)

    def leftgrad(self, first, second):
        return -0.5 * torch.zeros(first.natoms, 3).index_add(0, first.i,
                                                             (self.kern.leftgrad(first.d, second.d)[:, None]
                                                              * first.dr[..., None]).sum(dim=-1)
                                                             ).reshape(first.natoms*3, 1)

    def rightgrad(self, first, second):
        return -0.5 * torch.zeros(second.natoms, 3).index_add(0, second.i,
                                                              (self.kern.rightgrad(first.d, second.d)[..., None]
                                                               * second.dr).sum(dim=0)
                                                              ).reshape(1, second.natoms*3)

    def gradgrad(self, first, second):
        return torch.zeros(first.natoms, second.natoms, 3, 3
                           ).index_add(1, second.i, torch.zeros(first.natoms, second.i.size(0), 3, 3
                                                                ).index_add(0, first.i,
                                                                            (first.dr[:, None, :, None]
                                                                             * second.dr[None, :, None, :]
                                                                             * self.kern.gradgrad(
                                                                                 first.d, second.d)
                                                                             [..., None, None])
                                                                            )
                                       ).permute(0, 2, 1, 3).reshape(first.natoms*3, second.natoms*3)

