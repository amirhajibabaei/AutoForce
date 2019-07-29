
# coding: utf-8

# In[ ]:


from torch.nn import Module
from torch import cat
from theforce.util.util import iterable


class SimilarityKernel(Module):

    def __init__(self, kernel):
        super().__init__()
        self.kern = kernel
        self.params = self.kern.params

    def forward(self, first, second, operation='func'):
        return cat([cat([getattr(self, operation)(a, b) for a in iterable(first)], dim=0
                        ) for b in iterable(second)], dim=1)

    def diag(self, first, operation='func'):
        return cat([getattr(self, operation+'diag')(a) for a in iterable(first)])

    def funcdiag(self, first):
        return self.func(first, first).view(1)

    def func(self, p, q):
        return self.get_func(p, q)

    def leftgrad(self, p, q):
        return self.get_leftgrad(p, q)

    def rightgrad(self, p, q):
        return self.get_rightgrad(p, q)

    def gradgrad(self, p, q):
        return self.get_gradgrad(p, q)

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

