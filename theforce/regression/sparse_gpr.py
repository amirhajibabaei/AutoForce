
# coding: utf-8

# In[ ]:


"""
Two classes are defined: GPR and SparseGPR.
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


class GPR(Module):

    def __init__(self, X, Z, Y, chunks=None, derivatives=None):
        super(GPR, self).__init__()  # TODO: if Z is None: do full GPR

        self.X = X

        # values
        if Y is not None:
            self.chunks = chunks
            if chunks:  # TODO: generalize Const mean to more general types
                _s = torch.as_tensor(chunks).type(Y.dtype)
                self.mean = (Y/_s).mean()
            else:
                _s = 1
                self.mean = Y.mean()
            data = [Y - self.mean*_s]
            self.use_values = True
        else:
            self.mean = 0
            data = []
            self.use_values = False

        # derivatives
        if derivatives is not None:
            self.dX_dt = torch.as_tensor(derivatives[0])
            data += [torch.as_tensor(derivatives[1]).view(-1)]
            self.use_derivatives = True
        else:
            self.use_derivatives = False

        # parameters
        self._noise = Parameter(free_form(torch.tensor(1., dtype=X.dtype)))
        self.Z = Z       # Parameter or not is controled from SparseGPR
        self.kern = RBF(torch.ones_like(X[0]), torch.tensor(1., dtype=X.dtype))

        # combine values and derivatives:
        self.Y = torch.cat(data)
        self.zeros = torch.zeros_like(self.Y)
        self.ones = torch.ones_like(self.Y)

    def extra_repr(self):
        print('\nSGPR:\nnoise: {}'.format(positive(self._noise)))
        print('mean function used: constant {}\n'.format(self.mean))

    # --------------------------------------------------------------------
    def covariances(self):
        ZZ = self.kern.cov_matrix(self.Z, self.Z)

        # values
        if self.use_values:
            ZX = self.kern.cov_matrix(self.Z, self.X)
            if self.chunks:
                ZX = sum_packed_dim(ZX, self.chunks)
                tr = torch.stack([self.kern.cov_matrix(x, x).sum()
                                  for x in torch.split(self.X, self.chunks)]).sum()
            else:
                tr = self.X.size()[0]*self.kern.diag()
        else:
            tr = 0
            ZX = None

        # derivatives
        if self.use_derivatives:
            dZX = self.kern.cov_matrix(self.Z, self.X, d_dtheta=self.dX_dt,
                                       wrt=1).view(self.Z.size()[0], -1)
            tr = tr + self.kern.diag_derivatives(self.dX_dt)
            if ZX is None:
                ZX = dZX
            else:
                ZX = torch.cat((ZX, dZX), dim=1)
        return ZZ, ZX, tr

    def forward(self):

        # covariances
        ZZ, ZX, tr = self.covariances()
        noise = positive(self._noise)

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

        ZZ, ZX, _ = self.covariances()
        XZ = ZX.t()
        noise = positive(self._noise)

        # numerically stable calculation of _mu
        L, ridge = jitcholesky(ZZ, jitbase=2)
        A = torch.cat((XZ, noise * L.t()))
        Y = torch.cat((self.Y, torch.zeros(self.Z.size()[0],
                                           dtype=self.Y.dtype)))
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
        if x is None:
            return x
        else:
            if array:
                return x.detach().numpy()
            else:
                return x

    def predict(self, X, var=False, array=True, derivative=None):
        if not hasattr(self, 'ready') or not self.ready:
            self.evaluate()
        _X = torch.as_tensor(X)

        # predictive mean
        XZ = self.kern.cov_matrix(_X, self.Z)
        mu = self.mean + torch.mv(XZ, self._mu)

        # predictive variance
        if var:
            sig = torch.ones(_X.size()[0], dtype=self.X.dtype)*self.kern.diag() +                 torch.mm(XZ, torch.mm(self._sig, XZ.t())).diag()
            if (sig < 0).any():
                sig = torch.clamp(sig, 0)
                warnings.warn(
                    'variance clamped! variance is not numerically stable yet!')
        else:
            sig = None

        # derivative
        if derivative is not None:
            dXZ = self.kern.cov_matrix(_X, self.Z, d_dtheta=derivative)
            deriv = torch.einsum('ij...,j->i...', dXZ, self._mu)
        else:
            deriv = None

        return (self.out(out, array=array) for out in (mu, sig, deriv))

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
        print('trained for {} steps'.format(steps))

        self.ready = 0


class SparseGPR(GPR):

    def __init__(self, X, Y, num_inducing, chunks=None, derivatives=None):
        Z = Parameter(X[torch.randint(X.size()[0], (num_inducing,))])
        super(SparseGPR, self).__init__(X, Z, Y, chunks=chunks,
                                        derivatives=derivatives)

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
    chunks = torch.randint(1, 10, (100,)).tolist()
    X = torch.cat([(torch.rand(size, 1)-0.5)*5 for size in chunks])
    Y = (X.tanh() * (-X**2).exp()).view(-1) + 1 * 10.

    # transorm Y -> YY, trans
    YY = sum_packed_dim(Y, chunks)

    # define model
    #model = SparseGPR(X, Y, 6, None)
    model = SparseGPR(X, YY, 6, chunks)

    # training
    model.train(100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(model.losses)

    # predict
    with torch.no_grad():
        ax2.scatter(X, Y)
        Xtest = torch.arange(-3, 3, 0.1, dtype=X.dtype).view(-1, 1)
        Ytest, var, deriv = model.predict(Xtest, var=True, array=False)
        x, y, err = Xtest.numpy().reshape(-1), Ytest.numpy(), torch.sqrt(var).numpy()*10
        ax2.plot(x, y, color='green')
        ax2.fill_between(x, y-err, y+err, alpha=0.2)
        Z = model.Z.detach().numpy().reshape(-1)
        u = model.u.detach().numpy()
        ax2.scatter(Z, u, marker='x', s=200, color='red')

        Ytest, _, _ = model.predict(Xtest, var=False)
        assert Ytest.__class__ == np.ndarray

    print(model)


def second_test():
    """
    Here we have a chain relationship as Y=Y(X(T)).
    The data we are given are (X, dX_dT, dY_dT).
    We want to regress the latent function Y = f(X).
    """
    import numpy as np
    from theforce.regression.algebra import _2pi
    import pylab as plt
    get_ipython().run_line_magic('matplotlib', 'inline')

    chunks = torch.randint(1, 10, (10,)).tolist()
    T = torch.cat([torch.rand(size, 1) for size in chunks])
    X = _2pi*torch.cos(_2pi*T)
    dX_dT = -_2pi**2*torch.sin(_2pi*T)
    Y = torch.squeeze(torch.sin(X))
    dY_dT = dX_dT*torch.cos(X)
    YY = sum_packed_dim(Y, chunks)

    # define model
    dX_dT = torch.unsqueeze(dX_dT, dim=-1)
    model = SparseGPR(X, YY, 6, chunks, derivatives=(dX_dT, dY_dT))

    # return 0
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.scatter(X, dX_dT)
    ax2.scatter(X, dY_dT)
    ax3.scatter(X, Y)

    # training
    model.train(300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(model.losses)

    # predict
    with torch.no_grad():
        ax2.scatter(X, Y)
        #T = torch.arange(-1., 1., 0.01, dtype=X.dtype).view(-1, 1)
        #Xtest = _2pi*torch.cos(_2pi*T)
        Xtest = torch.arange(-_2pi, _2pi, 0.1, dtype=X.dtype).view(-1, 1)
        Ytest, var, _ = model.predict(Xtest, var=True, array=False)
        x, y, err = Xtest.numpy().reshape(-1), Ytest.numpy(), torch.sqrt(var).numpy()*10
        ax2.plot(x, y, color='green')
        ax2.fill_between(x, y-err, y+err, alpha=0.2)
        Z = model.Z.detach().numpy().reshape(-1)
        u = model.u.detach().numpy()
        ax2.scatter(Z, u, marker='x', s=200, color='red')

        Ytest, _, _ = model.predict(Xtest, var=False)
        assert Ytest.__class__ == np.ndarray

    print(model)


if __name__ == '__main__':

    # test_if_works()

    second_test()

