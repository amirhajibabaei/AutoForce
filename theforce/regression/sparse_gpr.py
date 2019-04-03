
# coding: utf-8

# In[ ]:


from math import pi
import torch
from torch.nn import Module, Parameter
from theforce.regression.algebra import positive, free_form
from theforce.regression.kernels import RBF
twopi = torch.tensor(2*pi)


class SGPR(Module):
    """
    Parameters: .Z, .noise, .kern.scale, .kern.variance
    """

    def __init__(self, X, Y, num_inducing, eps=1e-6):
        super(SGPR, self).__init__()

        self.n, self.d = X.size()
        self.m = num_inducing
        self.mean = Y.mean()
        self.X = X
        self.Y = Y - self.mean
        self.const = (self.Y**2).sum()

        # parameters
        self.Z = Parameter(X[torch.randint(self.n, (num_inducing,))])
        self._noise = Parameter(free_form(torch.tensor(1.)))
        self.kern = RBF(torch.ones(self.d), torch.tensor(1.))
        self.eps = eps

        self.ready = False

    def extra_repr(self):
        print('SGP:\nnoise: {}'.format(positive(self._noise)))
        print('Z:\n', self.Z.data)

    def shift(self, s):
        c = s.mean() * self.eps
        a = s.min() - c
        if a >= 0:
            return s
        else:
            return s - a

    def forward(self):
        """
        Returns negative of the (variational) Evidence Lower Bound (ELBO).
        """
        self.ready = False
        noise = positive(self._noise)
        ZZ = self.kern.cov_matrix(self.Z, self.Z)
        ZX = self.kern.cov_matrix(self.Z, self.X)
        alpha = torch.mv(ZX, self.Y)
        SIGMA = ZZ + torch.mm(ZX, ZX.t()) / noise**2
        #
        U, S, V = torch.svd(SIGMA)
        U_SIGMA = torch.mv(U.t(), alpha)
        V_SIGMA = torch.mv(V.t(), alpha)
        S_SIGMA = self.shift(S)
        B_SIGMA = U_SIGMA * V_SIGMA / S_SIGMA
        #
        U, S, V = torch.svd(ZZ)
        U_ZZ = torch.mm(U.t(), ZX)
        V_ZZ = torch.mm(V.t(), ZX)
        S_ZZ = self.shift(S)
        B_ZZ = torch.einsum('ij,i,ij', V_ZZ, 1./S_ZZ, U_ZZ)
        #
        F = self.const / noise**2 - (B_SIGMA).sum() / noise**4 +             S_SIGMA.log().sum() - S_ZZ.log().sum() +             self.n*torch.log(twopi*noise**2) +             (self.n*self.kern.diag() - B_ZZ) / noise**2
        return F/2

    def evaluate(self):
        """
        Calculates some matrices usefull for predictions.
        """
        noise = positive(self._noise)
        ZZ = self.kern.cov_matrix(self.Z, self.Z)
        ZX = self.kern.cov_matrix(self.Z, self.X)
        alpha = torch.mv(ZX, self.Y)
        SIGMA = ZZ + torch.mm(ZX, ZX.t()) / noise**2
        #
        U, S, V = torch.svd(SIGMA)
        S = self.shift(S)
        SIGMA_inv = torch.mm(V, torch.mm(torch.diag(1./S), U.t()))
        #
        U, S, V = torch.svd(ZZ)
        S = self.shift(S)
        ZZ_inv = torch.mm(V, torch.mm(torch.diag(1./S), U.t()))
        #
        self._mu = torch.mv(SIGMA_inv, alpha) / noise**2
        self._sig = SIGMA_inv - ZZ_inv
        #
        self.u = self.mean + torch.mv(ZZ, self._mu)
        self.ready = True

    def predict(self, X, covar=True):
        if not self.ready:
            self.evaluate()
        XZ = self.kern.cov_matrix(X, self.Z)
        mu = self.mean + torch.mv(XZ, self._mu)
        if covar:
            XX = self.kern.cov_matrix(X, X)
            sig = XX + torch.mm(XZ, torch.mm(self._sig, XZ.t()))
            return mu, sig
        else:
            return mu
        
        
def test_if_works():
    import pylab as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    X = (torch.rand(20,1)-0.5)*5
    Y = (X.tanh() * (-X**2).exp()).view(-1)
    model = SGPR(X, Y, 10)

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
        Xtest = torch.arange(-3,3,0.1, dtype=X.dtype).view(-1,1)
        Ytest, Cov = model.predict(Xtest)
        x, y, err = Xtest.numpy().reshape(-1), Ytest.numpy(), torch.sqrt(Cov.diag()).numpy()*10
        plt.plot(x, y, color='red')
        plt.fill_between(x, y-err, y+err, alpha=0.2)
        
        
if __name__=='__main__':
    test_if_works()

