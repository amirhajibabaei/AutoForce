
# coding: utf-8

# In[ ]:


from math import pi
import torch
_2pi = torch.tensor(2*pi)


def positive(x):
    return torch.log(1. + torch.exp(x))


def free_form(x):
    return torch.log(torch.exp(x) - 1.)


def cholesky(A, jit=1e-6):
    ridge = 0
    try:
        L = torch.cholesky(A)
    except RuntimeError:
        scale = A.diag().mean()
        ridge = jit*scale
        done = False
        while not done:
            try:
                L = torch.cholesky(A + ridge*torch.eye(
                    *A.size(), dtype=A.dtype))
                done = True
            except RuntimeError:
                ridge *= 10
            if ridge > scale:
                raise RuntimeError('cholesky was not successful!')
    return L, ridge


def scale_tril(K, Y, logdet=False, solve=False):
    """
    Inputs: Y, K
    K: a symmetric positive definite (covariance) matrix,
       this will be factored  as K = L @ L.t()
    Y: 1D or 2D
    Returns: Q, logdet, ridge
    ------------------------------------------------------
    The following equality holds:
    Q.t() @ Q = Y.t() @ K.inverse() @ Y
    """
    L, ridge = cholesky(K)
    if len(Y.size()) == 1:
        _1d, _Y = True, Y.view(-1, 1)
    else:
        _1d, _Y = False, Y
    if solve:
        Q, _ = torch.trtrs(_Y, L, upper=False)
    else:
        Q = torch.mm(L.inverse(), _Y)
    if logdet:
        ld = 2*L.diag().log().sum()
    else:
        ld = None
    return Q, ld, ridge


def log_normal(Y, K):
    Q, logdet, ridge = scale_tril(K, Y, logdet=True)
    return -(torch.mm(Q.t(), Q) + logdet + torch.log(_2pi)*Y.size()[0])/2


# tests ------------------------------------------------------------------------
def test(n=1000):

    # test cholesky
    K = torch.ones(n, n)
    L, ridge = cholesky(K)
    test_cholesky = torch.allclose(torch.mm(L, L.t()), K)
    print('Cholesky: {}'.format(test_cholesky))

    # test log_normal
    Y = torch.ones(n)
    dist = torch.distributions.MultivariateNormal(torch.zeros(n), scale_tril=L)
    test_log_normal = torch.allclose(dist.log_prob(Y), log_normal(Y, K))
    print('log_normal: {}'.format(test_log_normal))


if __name__ == '__main__':
    test()

