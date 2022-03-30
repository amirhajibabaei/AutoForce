# +
import autoforce.cfg as cfg
from autoforce.core import Function
from math import sqrt
import torch


class Harmonics(Function):

    """
    Description:

    * An implementation of (solid) spherical harmonics.
      It maps 3D Cartesian vectors as follows:

        r -> |r|^l Ylm(theta, phi)

      where Ylm is spherical harmonics function.

    * If |r| = 1, this reduces to spherical harmonics.

    * If the results are multiplied by sqrt(4*pi/(2*l+1)),
      then regular solid harmonics are obtained:

        https://en.wikipedia.org/wiki/Solid_harmonics


    * The results are stored in a matrix Y (per-input)
      such that the full harmonic with (l, m) can be
      retrieved from:

        m > 0 -> Y[l,l-m] + 1.0j*Y[l-m,l]
        m = 0 -> Y[l,l]

      where:
        l = 0, ..., lmax 
        m = 0, ..., l

    * With lmax = 3 this arrangement looks like:

            0 1 2 3       0 1 2 3        r i i i
        l = 1 1 2 3   m = 1 0 1 2    Y = r r i i
            2 2 2 3       2 1 0 1        r r r i
            3 3 3 3       3 2 1 0        r r r r

      where r, i indicate real and imaginary components.

    * For further clarification, see the "_scipy_harmonics"
      function in the same module.

    """

    def __init__(self, lmax: int) -> None:
        super().__init__()
        self.lmax = lmax
        self.Yoo = torch.sqrt(1./(4*cfg.pi))
        self._al, self._bl, self._cl, self._dl = _alp(lmax)
        self.l, self.m, self.sign = _l_m_s(lmax)
        _c = (self.l**2-self.m**2) * (2*self.l+1) / (2*self.l-1)
        self.coef = _c[1:, 1:].sqrt()

    def forward(self, rij: torch.Tensor) -> torch.Tensor:
        """
        The input should be a Tensor with shape [:, 3].

        """

        # 1.
        x, y, z = rij.T
        rxy2 = x*x + y*y
        r2 = rxy2 + z*z
        rxy = rxy2.add(cfg.eps).sqrt()
        r = r2.sqrt()

        # 2. Associated Legendre Polynomials
        alp = [[torch.full_like(r, self.Yoo)]]
        for l in range(1, self.lmax+1):
            alp.append([*(self._al[l][m]*(z*alp[l-1][m] +
                                          r2*self._bl[l][m]*alp[l-2][m])
                          for m in range(l-1)),
                        self._cl[l]*z*alp[l-1][l-1],
                        self._dl[l]*rxy*alp[l-1][l-1]
                        ])

        # 3. Sin & Cos of m*phi
        pole = rxy < cfg.eps
        sin_phi = torch.where(pole, y, y/rxy)
        cos_phi = torch.where(pole, x, x/rxy)
        sin = [torch.zeros_like(sin_phi), sin_phi]
        cos = [torch.ones_like(cos_phi), cos_phi]
        for m in range(2, self.lmax+1):
            s = sin_phi*cos[-1] + cos_phi*sin[-1]
            c = cos_phi*cos[-1] - sin_phi*sin[-1]
            sin += [s]
            cos += [c]

        # 4. Spherical Harmonics
        Y = torch.zeros(self.lmax+1, self.lmax+1, r.size(0), dtype=cfg.float_t)
        for l in range(self.lmax+1):
            Y[l, l] = alp[l][0]
            for m in range(l, 0, -1):
                Y[l, l-m] = alp[l][m]*cos[m]
                Y[l-m, l] = alp[l][m]*sin[m]

        return Y


def _alp(lmax: int) -> (list, list, list, list):
    """
    (l, m)-dependent constants needed for 
    calculating associated Legendre polynomials.

    """
    a = [None, None]
    b = [None, None]
    c = [torch.tensor(1.0, dtype=cfg.float_t),
         torch.tensor(sqrt(3), dtype=cfg.float_t)]
    d = [None, torch.tensor(-sqrt(1.5), dtype=cfg.float_t)]
    for l in range(2, lmax+1):
        _a = []
        _b = []
        for m in range(l-1):
            _a.append(sqrt((4*l**2-1)/(l**2-m**2)))
            _b.append(-sqrt(((l-1)**2-m**2)/(4*(l-1)**2-1)))
        _a = torch.tensor(_a, dtype=cfg.float_t).reshape(-1, 1)
        _b = torch.tensor(_b, dtype=cfg.float_t).reshape(-1, 1)
        _c = torch.tensor(sqrt(2*l+1), dtype=cfg.float_t)
        _d = torch.tensor(-sqrt((1+1/(2*l))), dtype=cfg.float_t)
        a.append(_a)
        b.append(_b)
        c.append(_c)
        d.append(_d)
    return a, b, c, d


def _l_m_s(lmax: int) -> (torch.Tensor, torch.Tensor):
    """
    Auxiliary function for (l, m) tables.

    """
    l = torch.empty(lmax+1, lmax+1, 1, dtype=cfg.float_t)
    m = torch.empty(lmax+1, lmax+1, 1, dtype=cfg.float_t)
    s = torch.empty(lmax+1, lmax+1, 1, dtype=cfg.float_t)
    for i in range(lmax+1):
        l[:i, i] = i
        l[i, :i+1] = i
        for j in range(lmax-i+1):
            m[i, i+j] = j
            m[i+j, i] = j
        s[i, :i] = -1
        s[i, i:] = 1
    return l, m, s
