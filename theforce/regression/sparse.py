from torch.distributions import LowRankMultivariateNormal, MultivariateNormal
import torch


class Titias2009:
    def __init__(self, K, y, sigma=0.01, method="backward"):
        """
        K       spd matrix (nxn)
        y       vector (n)
        sigma   noise
        method  "forward" or "backward" (can be changed on the fly)

        calculates: self.active

        After a number of sweeps, self.active (boolean mask) indicates
        the selected (inducing) rows/cols of K.

        mask: hidden variable
        If method is "forward", active = mask else (if "backward") active = ~mask.

        example:
        >>> b = Titias2009(K, y, 0.01, 'backward')
        >>> b.doit(1e-3)
        """
        self.K = K
        self.y = y
        self.sigma = torch.as_tensor(sigma)
        self.n = K.size(0)
        self._method = {"backward": 0, "forward": 1}[method]
        self._ub = None
        self.mask = torch.zeros(self.n).bool()
        self.tr = self.K.trace()
        self.hist = []

    @property
    def loc(self):
        return torch.zeros_like(self.K[0])

    @property
    def ones(self):
        return torch.ones_like(self.K[0])

    @property
    def eye(self):
        return torch.eye(self.n).type(self.K.type())

    @property
    def method(self):
        return {0: "backward", 1: "forward"}[self._method]

    @method.setter
    def method(self, value):
        _method = {"backward": 0, "forward": 1}[value]
        if _method != self._method:
            self._method = _method
            self.mask = ~self.mask

    @property
    def active(self):
        return self.mask if self._method else ~self.mask

    def mll(self):
        """Titsias lower-bound"""
        a = self.active
        q = (self.K[a][:, a].cholesky().inverse()@self.K[a]).T
        f = (LowRankMultivariateNormal(self.loc, q, self.sigma**2*self.ones).log_prob(self.y)
             - 0.5*(self.tr - (q*q).sum())/self.sigma**2)
        return f

    @property
    def upper_bound(self):
        if self._ub is None:
            self._ub = MultivariateNormal(self.loc, self.K+self.sigma**2*self.eye
                                          ).log_prob(self.y)
        return self._ub

    def projected(self):
        a = self.active
        L = self.K[a][:, a].cholesky()
        sigma = self.sigma
        A = torch.cat([self.K[:, a]/sigma.view(-1, 1), L.t()])
        O = torch.zeros(L.size(0)).type(L.type())
        Y = torch.cat([self.y/sigma, O])
        Q, R = torch.qr(A)
        mu = R.inverse()@Q.t()@Y
        return a, mu

    def step(self):
        indices = []
        mlls = []
        for i in range(self.n):
            if self.mask[i]:
                continue
            self.mask[i] = True
            indices += [i]
            mlls += [self.mll()]
            self.mask[i] = False
        if len(mlls) == 0:
            return 1. if self._method else 0.
        mlls = torch.stack(mlls)
        argmax = mlls.argmax()
        self.mask[indices[argmax]] = True
        self.hist += [mlls[argmax]]
        if len(self.hist) >= 2:
            delta = (self.hist[-1] - self.hist[-2])/abs(self.upper_bound)
        else:
            delta = 1. if self._method else 0.
        return delta

    def doit(self, alpha=1e-3, maxsteps=None, verbose=True):
        steps = 0
        with torch.no_grad():
            while True:
                steps += 1
                delta = self.step()
                if verbose:
                    print(f'{steps} {delta}')
                a = False if maxsteps is None else steps >= maxsteps
                b = abs(delta) < alpha if self._method else abs(delta) > alpha
                if a or b:
                    break
        a, mu = self.projected()
        return a, mu
