"""
Two classes are defined: GPR and SparseGPR.
The latter is a thin wrapper around the former.
Essentially the only responsibility of the latter is to
control the inducing data points.
They could be variational parameters, greedily selected,
or just simple (constant) tensors.
"""
import warnings

import torch
from torch.distributions import LowRankMultivariateNormal
from torch.nn import Module, Parameter

from theforce.regression.algebra import (
    free_form,
    jitcholesky,
    low_rank_factor,
    positive,
    sum_packed_dim,
)
from theforce.regression.kernels import RBF


class GPR(Module):
    def __init__(self, X, Z, Y, chunks=None, derivatives=None):
        super(GPR, self).__init__()  # TODO: if Z is None: do full GPR

        self.X = X

        # values
        if Y is not None:
            self.chunks = chunks
            if chunks:  # TODO: generalize Const mean to more general types
                _s = torch.as_tensor(chunks).type(Y.dtype)
                self.mean = (Y / _s).mean()
            else:
                _s = 1
                self.mean = Y.mean()
            data = [Y - self.mean * _s]
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
        self._noise = Parameter(free_form(torch.tensor(1.0, dtype=X.dtype)))
        self.Z = Z  # Parameter or not is controled from SparseGPR
        self.kern = RBF(torch.ones_like(X[0]), torch.tensor(1.0, dtype=X.dtype))

        # combine values and derivatives:
        self.Y = torch.cat(data)
        self.zeros = torch.zeros_like(self.Y)
        self.ones = torch.ones_like(self.Y)

    def extra_repr(self):
        print("\nSGPR:\nnoise: {}".format(positive(self._noise)))
        print("mean function used: constant {}\n".format(self.mean))

    # --------------------------------------------------------------------
    def covariances(self):
        ZZ = self.kern.cov_matrix(self.Z, self.Z)

        # values
        if self.use_values:
            ZX = self.kern.cov_matrix(self.Z, self.X)
            if self.chunks:
                ZX = sum_packed_dim(ZX, self.chunks)
                tr = torch.stack(
                    [
                        self.kern.cov_matrix(x, x).sum()
                        for x in torch.split(self.X, self.chunks)
                    ]
                ).sum()
            else:
                tr = self.X.size()[0] * self.kern.diag()
        else:
            tr = 0
            ZX = None

        # derivatives
        if self.use_derivatives:
            dZX = self.kern.cov_matrix(self.Z, self.X, d_dtheta=self.dX_dt, wrt=1).view(
                self.Z.size()[0], -1
            )
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
        trace = 0.5 * (tr - torch.einsum("ij,ij", Q, Q)) / noise**2

        # low rank MVN
        p = LowRankMultivariateNormal(self.zeros, Q.t(), self.ones * noise**2)

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
        Y = torch.cat((self.Y, torch.zeros(self.Z.size()[0], dtype=self.Y.dtype)))
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
        if not hasattr(self, "ready") or not self.ready:
            self.evaluate()
        _X = torch.as_tensor(X)

        # predictive mean
        XZ = self.kern.cov_matrix(_X, self.Z)
        mu = self.mean + torch.mv(XZ, self._mu)

        # predictive variance
        if var:
            sig = (
                torch.ones(_X.size()[0], dtype=self.X.dtype) * self.kern.diag()
                + torch.mm(XZ, torch.mm(self._sig, XZ.t())).diag()
            )
            if (sig < 0).any():
                sig = torch.clamp(sig, 0)
                warnings.warn(
                    "variance clamped! variance is not numerically stable yet!"
                )
        else:
            sig = None

        # derivative
        if derivative is not None:
            dXZ = self.kern.cov_matrix(_X, self.Z, d_dtheta=derivative)
            deriv = torch.einsum("ij...,j->i...", dXZ, self._mu)
        else:
            deriv = None

        return (self.out(out, array=array) for out in (mu, sig, deriv))

    # training -------------------------------------------------------------------
    def train(self, steps=100, optimizer=None, lr=0.1):

        if not hasattr(self, "losses"):
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
        print("trained for {} steps".format(steps))

        self.ready = 0


class SparseGPR(GPR):
    def __init__(self, X, Y, num_inducing, chunks=None, derivatives=None):
        Z = Parameter(X[torch.randint(X.size()[0], (num_inducing,))])
        super(SparseGPR, self).__init__(X, Z, Y, chunks=chunks, derivatives=derivatives)

    def extra_repr(self):
        super(SparseGPR, self).extra_repr()
        print("\nSparseGPR:\nZ:\n{}\n".format(self.Z.data))

    def pre_forward(self):
        pass

    def forward(self):
        self.pre_forward()
        return super(SparseGPR, self).forward()


# testing -------------------------------------------------------------------------
def plot_model(self, ax, ax2, Y):
    import numpy as np

    X = self.X
    pred_Y, var, _ = self.predict(X, var=True)
    ax.scatter(X, Y)
    ax.plot(self.out(X, array=True), pred_Y, color="red", lw=2)
    ax.scatter(
        self.out(self.Z, array=True), self.out(self.u, array=True), marker="X", s=200
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    err = np.sqrt(var) * 10
    ax2.fill_between(X.view(-1), pred_Y - err, pred_Y + err, alpha=0.2)
    ax.set_xlabel("X")
    ax.set_ylabel("10*sqrt(var)")


def plot_derivatives(self, ax, dX_dany, dY_dany, label="dY/d?"):
    X = self.X
    pred_Y, _, pred_d = self.predict(X, derivative=dX_dany)
    ax.scatter(X, dY_dany)
    ax.plot(self.out(X, array=True), pred_d, color="red", lw=2)
    ax.set_xlabel("X")
    ax.set_ylabel(label)


def test_if_works(data_function):
    def visualize_trained_model(model):
        import matplotlib.pylab as plt

        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        ax = axes[0]
        plot_model(model, axes[0], axes[1], Y)
        dX_dX = torch.ones_like(X)
        plot_derivatives(model, axes[2], dX_dX, dY_dX, "dY/dX")
        plot_derivatives(model, axes[3], dX_dT.view(-1, 1, 1), dY_dT, "dY/dT")
        fig.tight_layout()

    chunks = torch.randint(1, 10, (10,)).tolist()
    size = sum(chunks)
    T, X, dX_dT, Y, dY_dX, dY_dT = data_function(size=size)
    YY = sum_packed_dim(Y, chunks)

    X = X.view(-1, 1)
    dX_dX = torch.ones_like(X)
    dX_dT = dX_dT.view(-1, 1, 1)

    # select the data to be used
    # model = SparseGPR(X, Y, 8)
    # model = SparseGPR(X, YY, 8, chunks)
    # model = SparseGPR(X, None, 8, chunks, derivatives=(dX_dX, dY_dX))
    # model = SparseGPR(X, None, 8, chunks, derivatives=(dX_dT, dY_dT))
    model = SparseGPR(X, YY, 8, chunks, derivatives=(dX_dT, dY_dT))

    model.train(100)
    visualize_trained_model(model)


def dummy_data(size=100):
    T = torch.linspace(-1.0, 1.0, size)
    X = torch.exp(T) - torch.exp(-T)
    dX_dT = torch.exp(T) + torch.exp(-T)
    Y = torch.exp(-(X**2))
    dY_dX = -2 * X * Y
    dY_dT = dY_dX * dX_dT
    return T, X, dX_dT, Y, dY_dX, dY_dT


if __name__ == "__main__":

    test_if_works(dummy_data)
