
# coding: utf-8

# In[ ]:


"""
Two classes are defined: SGPR and SparseGPR.
The latter is a thin wrapper around the former.
Essentially the only responsibility of the latter is to
control the inducing data points.
They could be variational parameters, greedily selected,
or just simple (constant) tensors.
"""
import torch
from torch.nn import Module, Parameter
from torch.distributions import LowRankMultivariateNormal
from theforce.regression.algebra import positive, free_form, sum_packed_dim
from theforce.regression.algebra import low_rank_factor, jitcholesky
from theforce.regression.kernels import RBF
import warnings


class SGPR(Module):

    def __init__(self, X, Y, Z, sizes=None):
        super(SGPR, self).__init__()

        self.X = X
        self.sizes = sizes
        if sizes:
            _s = torch.as_tensor(sizes).type(Y.dtype)
            self.mean = (Y/_s).mean()
        else:
            _s = 1
            self.mean = Y.mean()
        self.Y = Y - self.mean*_s           # TODO: generalize Const mean to more types

        # parameters
        self._noise = Parameter(free_form(torch.tensor(1., dtype=X.dtype)))
        self.Z = Z       # Parameter or not is controled from SparseGPR
        self.kern = RBF(torch.ones_like(X[0]), torch.tensor(1., dtype=X.dtype))

        # zeros and ones
        self.zeros = torch.zeros_like(self.Y)
        self.ones = torch.ones_like(self.Y)

    def extra_repr(self):
        print('\nSGPR:\nnoise: {}'.format(positive(self._noise)))
        print('mean function used: constant {}\n'.format(self.mean))

    # --------------------------------------------------------------------
    def forward(self):
        noise = positive(self._noise)
        ZZ = self.kern.cov_matrix(self.Z, self.Z)
        ZX = self.kern.cov_matrix(self.Z, self.X)

        # if a transformation on ZX is needed or not # TODO: more efficient way?
        if self.sizes:
            ZX = sum_packed_dim(ZX, self.sizes)
            tr = torch.stack([self.kern.cov_matrix(x, x).sum()
                              for x in torch.split(self.X, self.sizes)]).sum()
        else:
            tr = self.X.size()[0]*self.kern.diag()

        # trace term
        Q, _, ridge = low_rank_factor(ZZ, ZX)
        trace = 0.5*(tr - torch.einsum('ij,ij', Q, Q))/noise**2

        # low rank MVN
        p = LowRankMultivariateNormal(self.zeros, Q.t(), self.ones*noise**2)

        # loss
        loss = -p.log_prob(self.Y) + trace
        return loss

    # ---------------------------------------------------------------------
    def evaluate(self):
        noise = positive(self._noise)
        ZZ = self.kern.cov_matrix(self.Z, self.Z)
        XZ = self.kern.cov_matrix(self.X, self.Z)
        if self.sizes:
            XZ = sum_packed_dim(XZ, self.sizes, dim=0)

        # numerically stable calculation of _mu
        L, ridge = jitcholesky(ZZ, jitbase=2)
        A = torch.cat((XZ, noise * L.t()))
        Y = torch.cat((self.Y, torch.zeros(self.Z.size()[0])))
        Q, R = torch.qr(A)
        self._mu = torch.mv(R.inverse(), torch.mv(Q.t(), Y))

        # inducing function values (Z, u)
        self.u = self.mean + torch.mv(ZZ, self._mu)

        # covariance ------------------------------ TODO: this is slightly ugly!
        ZZ_i = torch.mm(L.t().inverse(), L.inverse())
        SIGMA = ZZ + torch.mm(XZ.t(), XZ) / noise**2
        # SIGMA_i = SIGMA.inverse()
        Q, R = torch.qr(SIGMA)
        SIGMA_i = torch.mm(R.inverse(), Q.t())
        self._sig = SIGMA_i - ZZ_i
        # ------------------------------------------------------------------------

        self.ready = 1

    @staticmethod
    def out(x, array=False):
        if array:
            return x.detach().numpy()
        else:
            return x

    def predict(self, X, var=True, array=True):
        if not hasattr(self, 'ready') or not self.ready:
            self.evaluate()
        _X = torch.as_tensor(X)
        XZ = self.kern.cov_matrix(_X, self.Z)
        mu = self.mean + torch.mv(XZ, self._mu)
        if var:
            sig = torch.ones(_X.size()[0])*self.kern.diag() +                 torch.mm(XZ, torch.mm(self._sig, XZ.t())).diag()
            if (sig < 0).any():
                sig = torch.clamp(sig, 0)
                warnings.warn(
                    'variance clamped! variance is not numerically stable yet!')
            return self.out(mu, array=array), self.out(sig, array=array)
        else:
            return self.out(mu, array=array)

    # training -------------------------------------------------------------------
    def train(self, steps=100, optimizer=None, lr=0.1):

        if not hasattr(self, 'losses'):
            self.losses = []
            self.starts = []
        self.starts += [len(self.losses)]

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            loss = self.forward()
            self.losses += [loss.data]
            loss.backward()
            optimizer.step()
        print('trained for {} staps'.format(steps))


class SparseGPR(SGPR):

    def __init__(self, X, Y, num_inducing, sizes=None):
        Z = Parameter(X[torch.randint(Y.size()[0], (num_inducing,))])
        super(SparseGPR, self).__init__(X, Y, Z, sizes=sizes)

    def extra_repr(self):
        super(SparseGPR, self).extra_repr()
        print('\nSparseGPR:\nZ:\n{}\n'.format(self.Z.data))

    def pre_forward(self):
        pass

    def forward(self):
        self.pre_forward()
        return super(SparseGPR, self).forward()


# testing -------------------------------------------------------------------------
def test_if_works():
    import numpy as np
    import pylab as plt
    get_ipython().run_line_magic('matplotlib', 'inline')

    # dummy data
    sizes = torch.randint(1, 10, (100,)).tolist()
    X = torch.cat([(torch.rand(size, 1)-0.5)*5 for size in sizes])
    Y = (X.tanh() * (-X**2).exp()).view(-1) + 1 * 10.

    # transorm Y -> YY, trans
    YY = sum_packed_dim(Y, sizes)

    # define model
    #model = SparseGPR(X, Y, 6, None)
    model = SparseGPR(X, YY, 6, sizes)

    # training
    model.train(100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(model.losses)

    # predict
    with torch.no_grad():
        ax2.scatter(X, Y)
        Xtest = torch.arange(-3, 3, 0.1, dtype=X.dtype).view(-1, 1)
        Ytest, var = model.predict(Xtest, array=False)
        x, y, err = Xtest.numpy().reshape(-1), Ytest.numpy(), torch.sqrt(var).numpy()*10
        ax2.plot(x, y, color='green')
        ax2.fill_between(x, y-err, y+err, alpha=0.2)
        Z = model.Z.detach().numpy().reshape(-1)
        u = model.u.detach().numpy()
        ax2.scatter(Z, u, marker='x', s=200, color='red')

        Ytest = model.predict(Xtest, var=False)
        assert Ytest.__class__ == np.ndarray

    print(model)


if __name__ == '__main__':

    test_if_works()

