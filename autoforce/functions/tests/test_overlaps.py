# +
import autoforce.cfg as cfg
from autoforce.cfg import pi
from autoforce.functions import Overlaps, CosineCut
from autoforce.functions.coordinates import cartesian, rotate, rotation_matrix
import torch


def test_Overlaps_perm() -> bool:
    """
    Test if the descriptor is invariant wrt permutations.

    """

    # 1. Setup
    cutoff = 6.
    lmax = 4
    nmax = 4
    nj = 40
    type1 = 25
    type2 = nj - type1
    soap = Overlaps(lmax, nmax)
    cut = CosineCut()
    rij = torch.rand(nj, 3, dtype=cfg.float_t)
    rij.sub_(0.5).mul_(2*cutoff+1.)
    species = torch.tensor(type1*[1]+type2*[2])

    # 2. Test
    dij = rij.norm(dim=1)
    wj = cut.function(dij, cutoff)
    _, _, y = soap.function(rij, species, wj)
    perm = torch.randperm(nj)
    _, _, y2 = soap.function(rij[perm], species[perm], wj[perm])

    return y2.allclose(y)


def test_Overlaps_backward():
    """
    Test if backward runs without error.

    """

    # 1. Setup
    cutoff = 6.
    lmax = 4
    nmax = 4
    nj = 40
    type1 = 25
    type2 = nj - type1
    soap = Overlaps(lmax, nmax)
    cut = CosineCut()
    rij = torch.rand(nj, 3, dtype=cfg.float_t)
    rij.sub_(0.5).mul_(2*cutoff+1.)
    species = torch.tensor(type1*[1]+type2*[2])

    # 2. Test
    rij.requires_grad = True
    dij = rij.norm(dim=1)
    wj = cut.function(dij, cutoff)
    _, _, y = soap.function(rij, species, wj)
    y.sum().backward()

    return True


def test_Overlaps_rotational_invariance():
    """
    Test if Overlaps is rotationally invariant.

    """

    # 1. Setup
    cutoff = 6.
    lmax = 6
    nmax = 6
    nj = 40
    type1 = 25
    type2 = nj - type1
    soap = Overlaps(lmax, nmax)
    cut = CosineCut()
    species = torch.tensor(type1*[1]+type2*[2])
    r = torch.rand(nj, dtype=cfg.float_t)*cutoff + 0.5
    theta = torch.rand(nj, dtype=cfg.float_t)*pi
    phi = torch.rand(nj, dtype=cfg.float_t)*2*pi
    rij = cartesian(r, theta, phi)
    wj = cut.function(r, cutoff)
    R_z = rotation_matrix([0., 0., 1.], pi/18)
    R_y = rotation_matrix([0., 1., 0.], pi/18)

    # 2. Test
    y0 = None
    errors = []
    for _ in range(18):
        rij = rotate(rij, R_y)
        for __ in range(36):
            rij = rotate(rij, R_z)
            _, _, y = soap.function(rij, species, wj=wj)
            if y0 is None:
                y0 = y
                norm = y0.norm()
            else:
                errors.append((y-y0).norm()/norm)

    return max(errors)


def test_Overlaps_compressed_norm():
    """
    Test if the norm is the the same when
        1) compress = True
        2) compress = False

    """

    # 1. Setup
    cutoff = 6.
    lmax = 3
    nmax = 4
    nj = 40
    type1 = 25
    type2 = nj - type1
    soap = Overlaps(lmax, nmax)
    cut = CosineCut()
    rij = torch.rand(nj, 3, dtype=cfg.float_t)
    rij.sub_(0.5).mul_(2*cutoff+1.)
    species = torch.tensor(type1*[1]+type2*[2])

    # 2. Test
    rij.requires_grad = True
    dij = rij.norm(dim=1)
    wj = cut.function(dij, cutoff)
    _, _, y1 = soap.function(rij, species, wj, compress=True)
    _, _, y2 = soap.function(rij, species, wj, compress=False)

    return y1.norm().isclose(y2.norm())


if __name__ == '__main__':
    assert test_Overlaps_perm()
    assert test_Overlaps_backward()
    assert test_Overlaps_rotational_invariance() < 5e-5
    assert test_Overlaps_compressed_norm()
