
# coding: utf-8

# In[ ]:


from torch.nn import Module
from torch import cat
from theforce.regression.gp import Covariance
from theforce.util.util import iterable


class SimilarityKernel(Module):

    def __init__(self, kernels):
        super().__init__()
        self.kern = Covariance(kernels)
        self.params = self.kern.params

    def forward(self, first, second, operation='func'):
        return cat([cat([getattr(self, operation)(a, b) for a in iterable(first)], dim=0
                        ) for b in iterable(second)], dim=1)

    def diag(self, first, operation='func'):
        return cat([getattr(self, operation+'diag')(a) for a in iterable(first)])

    def funcdiag(self, first):
        return self.func(first, first).view(1)

    def func(self, p, q):
        raise NotImplementedError('func!')

    def leftgrad(self, p, q):
        raise NotImplementedError('leftgrad!')

    def rightgrad(self, p, q):
        raise NotImplementedError('rightgrad!')

    def gradgrad(self, p, q):
        raise NotImplementedError('gradgrad!')

    def gradgraddiag(self, p):
        raise NotImplementedError('gradgraddiag!')

    @property
    def state_args(self):
        return self.kern.state_args

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

