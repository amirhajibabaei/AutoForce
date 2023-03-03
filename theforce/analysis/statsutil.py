# +
import numpy as np
import torch


def moving_average(x, w, axis=0):
    a = np.cumsum(x, dtype=np.float, axis=axis)
    a[w:] = a[w:] - a[:-w]
    return a[w - 1 :] / w


class Cov_otf:
    """
    builds the covariance matrix on-the-fly.
    """

    def __init__(self):
        self.k = 0
        self.i = 0
        self.ij = 0
        self._eig = None

    def __call__(self, _y):
        if _y.dim() == 2:
            y = _y
        elif _y.dim() == 1:
            y = _y[:, None]
        else:
            raise RuntimeError("y.dim?")
        self.k += 1
        self.i += y
        self.ij += y * y.t()
        self._eig = None

    @property
    def mat(self):
        return self.ij / self.k - self.i * self.i.t() / self.k**2

    @property
    def eig(self):
        if self._eig is None:
            self._eig = torch.eig(self.mat, eigenvectors=True)
            assert self._eig.eigenvalues[:, 1].allclose(torch.zeros([]))  # real
        return self._eig

    @property
    def eigval(self):
        return self.eig.eigenvalues[:, 0]

    @property
    def eigvec(self):
        return self.eig.eigenvectors
