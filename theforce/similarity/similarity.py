#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.nn import Module
from torch import cat
from theforce.util.util import iterable
from theforce.util.caching import method_caching
from theforce.util.parallel import method_forker


class SimilarityKernel(Module):

    def __init__(self, kernel):
        super().__init__()
        self.kern = kernel
        self.params = self.kern.params

    @method_forker
    def forward(self, first, second, operation='func'):
        return cat([cat([getattr(self, operation)(a, b) for a in iterable(first)], dim=0
                        ) for b in iterable(second)], dim=1)

    @method_forker
    def diag(self, first, operation='func'):
        return cat([getattr(self, operation+'diag')(a) for a in iterable(first)])

    @method_forker
    def funcdiag(self, first):
        return self.func(first, first).view(1)

    @method_caching
    def func(self, p, q):
        return self.get_func(p, q)

    @method_caching
    def leftgrad(self, p, q):
        return self.get_leftgrad(p, q)

    @method_caching
    def rightgrad(self, p, q):
        return self.get_rightgrad(p, q)

    @method_caching
    def gradgrad(self, p, q):
        return self.get_gradgrad(p, q)

    @method_caching
    def gradgraddiag(self, p):
        return self.get_gradgraddiag(p)

    def save_for_later(self, loc, keyvals):
        for key, val in keyvals.items():
            setattr(loc, self.name+'_'+key, val)

    def saved(self, atoms_or_loc, key):
        return getattr(atoms_or_loc, self.name+'_'+key)

    @property
    def state_args(self):
        return self.kern.state

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

