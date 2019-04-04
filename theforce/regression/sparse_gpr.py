
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module, Parameter
from torch.distributions import LowRankMultivariateNormal
from theforce.regression.algebra import positive, free_form, low_rank_factor, cholesky
from theforce.regression.kernels import RBF
import warnings


class SGPR(Module):

    def __init__(self, X, Y, num_inducing):
        super(SGPR, self).__init__()

        self.n, self.d = X.size()
        self.m = num_inducing
        self.mean = Y.mean()
        self.X = X
        self.Y = Y - self.mean

        # parameters
        self.Z = Parameter(X[torch.randint(self.n, (num_inducing,))])
        self._noise = Parameter(free_form(torch.tensor(1.)))
        self.kern = RBF(torch.ones(self.d), torch.tensor(1.))

    def extra_repr(self):
        print('SGP:\nnoise: {}'.format(positive(self._noise)))
        print('Z:\n', self.Z.data)

    # --------------------------------------------------------------------

    def forward(self):
        noise = positive(self._noise)
        ZZ = self.kern.cov_matrix(self.Z, self.Z)
        ZX = self.kern.cov_matrix(self.Z, self.X)
        # trace term
        Q, _, ridge = low_rank_factor(ZZ, ZX)
        trace = 0.5*(self.n*self.kern.diag() -
                     torch.einsum('ij,ij', Q, Q))/noise**2
        # low rank log_prob
        loc = torch.zeros_like(self.Y)
        cov_factor = Q.t()
        cov_diag = torch.ones_like(self.Y) * noise**2
        p = LowRankMultivariateNormal(loc, cov_factor, cov_diag)
        # loss
        loss = -p.log_prob(self.Y) + trace
        return loss

    # ---------------------------------------------------------------------

    def evaluate(self):
        noise = positive(self._noise)
        ZZ = self.kern.cov_matrix(self.Z, self.Z)
        XZ = self.kern.cov_matrix(self.X, self.Z)

        # numerically stable calculation of _mu
        L, ridge = cholesky(ZZ, base=2)
        A = torch.cat((XZ, noise * L.t()))
        Y = torch.cat((self.Y, torch.zeros(self.m)))
        Q, R = torch.qr(A)
        self._mu = torch.mv(R.inverse(), torch.mv(Q.t(), Y))

        # inducing function values (Z, u)
        self.u = self.mean + torch.mv(ZZ, self._mu)

        # covariance ------------------------------ the last piece of the puzzle!
        # TODO: this is slightly ugly!
        ZZ_i = torch.mm(L.t().inverse(), L.inverse())
        SIGMA = ZZ + torch.mm(XZ.t(), XZ) / noise**2
        #SIGMA_i = SIGMA.inverse()
        Q, R = torch.qr(SIGMA)
        SIGMA_i = torch.mm(R.inverse(), Q.t())
        self._sig = SIGMA_i - ZZ_i
        # ------------------------------------------------------------------------

        self.ready = True

    def predict(self, X, var=True):
        if not hasattr(self, 'ready'):
            self.evaluate()
        _X = torch.as_tensor(X)
        XZ = self.kern.cov_matrix(_X, self.Z)
        mu = self.mean + torch.mv(XZ, self._mu)
        if var:
            sig = torch.ones(X.size()[0])*self.kern.diag() +                 torch.mm(XZ, torch.mm(self._sig, XZ.t())).diag()
            if (sig < 0).any():
                sig = torch.clamp(sig, 0)
                warnings.warn(
                    'variance clamped! variance is not numerically stable yet!')
            return mu, sig
        else:
            return mu


def test_if_works():
    import pylab as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    X = (torch.rand(100, 1)-0.5)*5
    Y = (X.tanh() * (-X**2).exp()).view(-1)
    model = SGPR(X, Y, 3)

    # optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for i in range(1):
        for _ in range(100):
            def closure():
                global training
                optimizer.zero_grad()
                loss = model.forward()
                loss.backward()
                return loss
            optimizer.step(closure)
        print((i+1)*100)
    model.evaluate()

    # predict
    with torch.no_grad():
        plt.scatter(X, Y)
        Xtest = torch.arange(-3, 3, 0.1, dtype=X.dtype).view(-1, 1)
        Ytest, var = model.predict(Xtest)
        x, y, err = Xtest.numpy().reshape(-1), Ytest.numpy(), torch.sqrt(var).numpy()*10
        plt.plot(x, y, color='green')
        plt.fill_between(x, y-err, y+err, alpha=0.2)
        Z = model.Z.detach().numpy().reshape(-1)
        u = model.u.detach().numpy()
        plt.scatter(Z, u, marker='x', s=200, color='red')


if __name__ == '__main__':

    test_if_works()

