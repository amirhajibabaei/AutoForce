
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module, Parameter
from theforce.regression.algebra import positive, free_form
import warnings


class RBF(Module):
    """
    Parameters: scale, variance
    """

    def __init__(self, scale, variance):
        super(RBF, self).__init__()
        self._scale = Parameter(free_form(scale))
        self._variance = Parameter(free_form(variance))

    def cov_matrix(self, x, xx, d_dtheta=None, wrt=0, sumd=True):
        """
        It returns K if d_dtheta=None.
        If d_dtheta is given it means that some derivative should be 
        calculated. "theta" could be x, xx, or sth else.
        If d_dtheta = ones_like(x) and wrt=0 (ones_like(xx) and wrt=1)
        it returns dK/dx (dK/dxx).
        If d_dtheta has a dimensionality greater than 2 it means that 
        it holds derivative of x (or xx depending on wrt) wrt sth else.
        Then, it will return dK/dtheta.
        It will sum over the features dimension if sumd=True.
        """
        assert wrt == 0 or wrt == 1
        scale = positive(self._scale)
        r = (x[:, None, ] - xx[None, ]) / scale
        cov = (-(r**2).sum(dim=-1)/2).exp() * self.diag()
        if d_dtheta is not None:
            r = r / scale
            for _ in range(d_dtheta.dim()-2):
                r = torch.unsqueeze(r, -1)
                cov = torch.unsqueeze(cov, -1)
            _derivative = (r*torch.unsqueeze(d_dtheta, dim=1-wrt)
                           ) * (-(-1)**(wrt))
            if sumd:
                cov = cov * _derivative.sum(dim=2)
            else:
                cov = torch.unsqueeze(cov, dim=2) * _derivative
        return cov

    def matrices(self, x, xx, d_dx=False, d_dxx=False, d_dxdxx=False):

        # covariance matrix
        iscale = 1.0/positive(self._scale)
        r = (x[:, None, ] - xx[None, ]) * iscale
        cov = (-(r**2).sum(dim=-1)/2).exp() * self.diag()

        # derivatives
        _dx = _dxx = _dxdxx = None
        if d_dx or d_dxx or d_dxdxx:
            rr = (r*iscale)
        if d_dx or d_dxx:
            cov_r = cov[..., None] * rr

        if d_dx:
            _dx = -cov_r
        if d_dxx:
            _dxx = cov_r
        if d_dxdxx:
            rirj = -rr[..., None] * rr[..., None, :]
            d = torch.LongTensor(torch.arange(iscale.size()[0]))
            rirj[..., d, d] = rirj[..., d, d] + iscale**2
            _dxdxx = rirj * cov[..., None, None]

        return cov, _dx, _dxx, _dxdxx

    def diag(self):
        return positive(self._variance)

    def diag_derivatives(self, d_dtheta):
        warnings.warn(
            'diag_derivatives in RBF is deprecated! It may be incorrect!')
        scale = positive(self._scale)
        for _ in range(d_dtheta.dim()-2):
            scale = torch.unsqueeze(scale, -1)
        return ((d_dtheta / scale)**2).sum() * self.diag()

    def extra_repr(self):
        print('\nRBF parameters: \nscale: {}\nvariance: {}\n'.format(
            positive(self._scale).data, positive(self._variance).data))


class Cauchy(Module):

    def __init__(self, scale, variance):
        super(Cauchy, self).__init__()
        self._scale = Parameter(free_form(scale))
        self._variance = Parameter(free_form(variance))

    def matrices(self, x, xx, d_dx=False, d_dxx=False, d_dxdxx=False):
        # covariance matrix
        scale = positive(self._scale)
        r = (x[:, None, ]-xx[None, ])/scale
        cov = 1.0/(1.0+(r**2).sum(dim=-1))

        # derivatives
        _dx = _dxx = _dxdxx = None
        if d_dx or d_dxx:
            a = (-2*r*cov[..., None]**2/scale) * self.diag()
            if d_dx:
                _dx = a
            if d_dxx:
                _dxx = -a

        if d_dxdxx:
            _dxdxx = -(8*r[..., None, :]*r[..., None]*cov[..., None, None] - 2*torch.eye(scale.size(0))
                       )*cov[..., None, None]**2/(scale[None, ]*scale[:, None]) * self.diag()
            
        cov = self.diag()*cov
        return cov, _dx, _dxx, _dxdxx

    def diag(self):
        return positive(self._variance)


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
    K = kern.cov_matrix(X1, X2, d_dtheta=dX, wrt=1, sumd=False)
    print(K.shape)
    dX = torch.ones_like(X1)
    K = kern.cov_matrix(X1, X2, d_dtheta=dX, wrt=0, sumd=False)
    print(K.shape)

    a, b, c, d = kern.matrices(X1, X2, True, True, True)
    print([p.shape for p in [a, b, c, d]])
    
    #--------------------------------------
    kern = Cauchy(scale, variance)
    a, b, c, d = kern.matrices(X1, X2, True, True, True)
    print([p.shape for p in [a, b, c, d]])

    

if __name__ == '__main__':
    test_if_works()

