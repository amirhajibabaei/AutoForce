
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

