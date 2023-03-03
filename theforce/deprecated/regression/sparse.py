# +
import torch
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal


class Titias2009:
    def __init__(self, K, y, sigma, method="forward"):
        """
        K       spd matrix (nxn)
        y       vector (n)
        sigma   noise (if noiseless, make it very small but nonzero, e.g 1e-4)
        method  "forward" or "backward" (can be changed on the fly)

        calculates: active, mu (mainly)

        After a number of sweeps, self.active (boolean mask) indicates
        the selected (inducing) rows/cols of K.

        mask: hidden variable
        If method is "forward", active = mask else (if "backward") active = ~mask.

        Main method is .doit(MAE, MAXAE).
        example:
        >>> b = Titias2009(K, y, 0.01, 'backward')
        >>> b.doit(0.01, 0.05)
        """
        self.K = K
        self.y = y.view(-1)
        self.sigma = torch.as_tensor(sigma)
        self.n = K.size(0)
        self._method = {"backward": 0, "forward": 1}[method]
        self.mask = torch.zeros(self.n).bool()
        self.diag = self.K.diag()
        self.hist = []
        self.hist_j = []

    @property
    def progress(self):
        return self.mask.long().sum()

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
        """Titsias's 2009 lower-bound"""
        a = self.active
        q = (self.K[a][:, a].cholesky().inverse() @ self.K[a]).T
        f = (
            LowRankMultivariateNormal(
                self.loc, q, self.sigma**2 * self.ones
            ).log_prob(self.y)
            - 0.5 * (self.diag[~a].sum() - (q[~a] ** 2).sum()) / self.sigma**2
        )
        return f

    def mll_upper_bound(self):
        mvn = MultivariateNormal(self.loc, self.K + self.sigma**2 * self.eye)
        return mvn.log_prob(self.y)

    def projected(self):
        a = self.active
        L = self.K[a][:, a].cholesky()
        sigma = self.sigma
        A = torch.cat([self.K[:, a] / sigma.view(-1, 1), L.t()])
        O = torch.zeros(L.size(0)).type(L.type())
        Y = torch.cat([self.y / sigma, O])
        Q, R = torch.qr(A)
        mu = R.inverse() @ Q.t() @ Y
        delta = ((self.K[:, a] @ mu).view(-1) - self.y).abs()
        return a, mu, delta.mean(), delta.max()

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
        if len(mlls) > 0:
            mlls = torch.stack(mlls)
            argmax = mlls.argmax()
            self.mask[indices[argmax]] = True
            self.hist += [mlls[argmax]]
            self.hist_j += [indices[argmax]]
        return self.projected()

    def doit(self, mae, maxae, maxsteps=None, verbose=True):
        """
        mae: mean absolute error
        maxae: max absolute error
        """
        if maxsteps is None:
            maxsteps = self.n - self.progress
        else:
            maxsteps = min(self.n - self.progress, maxsteps)
        if verbose:
            print(f"steps\tMAE\tMAXAE\t(maxsteps={maxsteps})")
        #
        active, mu, mean_ae, max_ae = self.projected()
        converged = (
            (mean_ae < mae and max_ae < maxae)
            if self._method
            else (mean_ae > mae or max_ae > maxae)
        )
        if verbose:
            print(f"0\t{mean_ae:.2g}\t{max_ae:.2g}")
        if converged:
            if verbose:
                print("already converged!")
            return active, mu
        #
        steps = 0
        with torch.no_grad():
            while True:
                steps += 1
                active, mu, mean_ae, max_ae = self.step()
                if verbose:
                    print(f"{steps}\t{mean_ae:.2g}\t{max_ae:.2g}")
                converged = (
                    (mean_ae < mae and max_ae < maxae)
                    if self._method
                    else (mean_ae > mae or max_ae > maxae)
                )
                if steps >= maxsteps or converged:
                    break
        return active, mu
