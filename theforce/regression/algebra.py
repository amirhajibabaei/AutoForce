
# coding: utf-8

# In[ ]:


from math import pi
import torch
_2pi = torch.tensor(2*pi)


# general ---------------------------------------
def positive(x):
    return torch.log(1. + torch.exp(x))


def free_form(x):
    return torch.log(torch.exp(x) - 1.)


def sum_packed_dim(packed, sizes, dim=-1):
    result = torch.stack([piece.sum(dim=dim)
                          for piece in torch.split(packed, sizes, dim=dim)], dim=dim)
    return result


# decompositions ---------------------------------------------
def jitcholesky(A, jit=1e-6, jitbase=2):
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
                ridge *= jitbase
            if ridge > scale:
                raise RuntimeError('cholesky was not successful!')
    return L, ridge


def low_rank_factor(K, Y, logdet=False, solve=False, jitbase=2):
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
    L, ridge = jitcholesky(K, jitbase=jitbase)
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
    Q, logdet, ridge = low_rank_factor(K, Y, logdet=True)
    return -(torch.mm(Q.t(), Q) + logdet + torch.log(_2pi)*Y.size()[0])/2


# examples ---------------------------------------------------------------------
def example_sum_packed_dim():
    sizes = torch.randint(1, 10, (100,))
    Y = [torch.ones(7, size) for size in sizes]
    Y = torch.cat(Y, dim=1)
    P = sum_packed_dim(Y, sizes.tolist())
    #print(Y.size(), P.size())
    print('sum_packed_dim works: {}'.format(P.size() == torch.Size([7, 100])))


# tests ------------------------------------------------------------------------
def test(n=1000):

    # test cholesky
    K = torch.ones(n, n)
    L, ridge = jitcholesky(K)
    test_cholesky = torch.allclose(torch.mm(L, L.t()), K)
    print('Cholesky: {}'.format(test_cholesky))

    # test log_normal
    Y = torch.ones(n)
    dist = torch.distributions.MultivariateNormal(torch.zeros(n), scale_tril=L)
    test_log_normal = torch.allclose(dist.log_prob(Y), log_normal(Y, K))
    print('log_normal: {}'.format(test_log_normal))


if __name__ == '__main__':
    example_sum_packed_dim()
    test()
