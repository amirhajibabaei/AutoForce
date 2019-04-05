
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module, Parameter
from theforce.regression.algebra import positive, free_form


class RBF(Module):
    """
    Parameters: scale, variance
    """

    def __init__(self, scale, variance):
        super(RBF, self).__init__()
        self._scale = Parameter(free_form(scale))
        self._variance = Parameter(free_form(variance))

    def cov_matrix(self, x, xx):
        r = (x[:, None, ] - xx[None, ]) / positive(self._scale)
        cov = (-(r**2).sum(dim=-1)/2).exp() * self.diag()
        return cov

    def diag(self):
        return positive(self._variance)

    def extra_repr(self):
        print('\nRBF parameters: \nscale: {}\nvariance: {}\n'.format(
            positive(self._scale).data, positive(self._variance).data))


def test_if_works():
    from theforce.regression.algebra import cholesky
    variance = torch.tensor(1.0)
    scale = torch.ones(7)
    kern = RBF(scale, variance)
    X = torch.rand(1000, 7)
    K = kern.cov_matrix(X, X)
    L, ridge = cholesky(K)
    print('ridge for cholesky decomposition: {}\n'.format(ridge))
    print(kern)


if __name__ == '__main__':
    test_if_works()

