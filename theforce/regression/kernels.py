
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

    def cov_matrix(self, x, xx, d_dtheta=None, wrt=0):
        assert wrt == 0 or wrt == 1
        scale = positive(self._scale)
        r = (x[:, None, ] - xx[None, ]) / scale
        cov = (-(r**2).sum(dim=-1)/2).exp() * self.diag()
        if d_dtheta is not None:
            r = r / scale
            for _ in range(d_dtheta.dim()-2):
                r = torch.unsqueeze(r, -1)
                cov = torch.unsqueeze(cov, -1)
            cov = cov * (r*torch.unsqueeze(d_dtheta, dim=1-wrt)
                         ).sum(dim=2) * (-(-1)**(wrt))          # NOTE: (-1)**wrt
        return cov

    def diag(self):
        return positive(self._variance)

    def extra_repr(self):
        print('\nRBF parameters: \nscale: {}\nvariance: {}\n'.format(
            positive(self._scale).data, positive(self._variance).data))


def test_if_works():
    from theforce.regression.algebra import jitcholesky
    variance = torch.tensor(1.0)
    scale = torch.ones(7)
    kern = RBF(scale, variance)
    X = torch.rand(1000, 7)
    K = kern.cov_matrix(X, X)
    L, ridge = jitcholesky(K)
    print('ridge for cholesky decomposition: {}\n'.format(ridge))
    print(kern, '\n')

    #
    X1 = torch.rand(19, 7)
    X2 = torch.rand(33, 7)
    dX = torch.rand(33, 7, 3)
    K = kern.cov_matrix(X1, X2, d_dtheta=dX, wrt=1)
    print(K.shape)


if __name__ == '__main__':
    test_if_works()

