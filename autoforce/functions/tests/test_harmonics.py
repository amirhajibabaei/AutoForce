# +
import autoforce.cfg as cfg
from autoforce.functions import Harmonics
from autoforce.functions.coordinates import r_theta_phi
from scipy.special import sph_harm
import torch


def _scipy_harmonics(rij: torch.Tensor, lmax: int) -> torch.Tensor:
    """
    Same functionality as Harmonics implemented
    with scipy for testing.

    """

    r, theta, phi = r_theta_phi(rij)
    rlm = torch.empty((lmax+1, lmax+1, rij.shape[0]))
    for l in range(0, lmax+1):
        for m in range(0, l+1):
            val = r**l*sph_harm(m, l, phi, theta)
            rlm[l, l-m] = val.real
            if m > 0:
                rlm[l-m, l] = val.imag

    return rlm


def test_Harmonics_scipy(lmax: int = 10) -> float:
    """
    Test if "Harmonics" is consistent scipy.

    Observed error: 2.5e-7

    """

    rlm = Harmonics(lmax)

    x = torch.tensor([[1., 0., 0.]], dtype=cfg.float_t)
    y = torch.tensor([[0., 1., 0.]], dtype=cfg.float_t)
    z = torch.tensor([[0., 0., 1.]], dtype=cfg.float_t)

    error = []
    for r in [x, y, z]:
        a = _scipy_harmonics(r, lmax)
        b = rlm.function(r)
        error.append((a-b).abs().max())

    return float(max(error))


def test_Harmonics_rotational_invariance(lmax: int = 10,
                                         size: int = 1000
                                         ) -> (float, float):
    """
    Test if "Harmonics" satisfies a known rotational
    invariance equation for spherical harmonics.

    Also check if autograd gradients are numerically stable
    at all polar angles.

    Observed error: 3e-7

    Observed grad error: 1e-6

    """

    # unit vectors in x-z plane
    theta = torch.linspace(0, cfg.pi, size, dtype=cfg.float_t)
    x = theta.sin()
    y = torch.zeros_like(x)
    z = theta.cos()
    xyz = torch.stack((x, y, z), dim=1)

    # harmonics
    lmax = 3
    rlm = Harmonics(lmax)
    xyz.grad = None
    xyz.requires_grad = True
    y = rlm.function(xyz)

    # invariant parameter
    a = 2*torch.ones(lmax+1, lmax+1, dtype=cfg.float_t)
    a -= torch.eye(lmax+1, dtype=cfg.float_t)
    a.unsqueeze_(-1)
    a /= 2*rlm.l+1
    a *= 4*cfg.pi/(lmax+1)
    _1 = (a*y.pow(2)).sum(dim=(0, 1))

    # errors
    error = _1.sub(1.).abs().max().detach()
    _1.sum().backward()
    grad_error = xyz.grad.norm(dim=1).sub(3).abs().max()

    return float(error), float(grad_error)


if __name__ == '__main__':
    test_Harmonics_scipy()
    test_Harmonics_rotational_invariance()
