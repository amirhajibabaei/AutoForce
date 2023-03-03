# +
from math import pi

import torch

from theforce.descriptor.cutoff import PolyCut
from theforce.descriptor.ylm import Ylm


class Ql:
    def __init__(self, lmax, cutoff):
        self.ylm = Ylm(lmax)
        self.radial = PolyCut(cutoff)
        one = torch.ones(lmax + 1, lmax + 1)
        self.Yr = 2 * torch.tril(one) - torch.eye(lmax + 1)
        self.Yi = 2 * torch.triu(one, diagonal=1)
        self.coeff = 4 * pi / torch.arange(0, lmax + 1).mul(2).add(1.0)

    def __call__(self, xyz):
        r = xyz.pow(2).sum(dim=-1).sqrt()
        select = r < self.radial.rc
        xyz = xyz[select]
        r = r[select]
        rad = self.radial(r, grad=False)
        y = self.ylm(xyz, with_r=1.0, grad=False)
        qlm = ((y * rad).sum(dim=-1) / rad.sum()) ** 2
        ql = (self.Yr * qlm).sum(dim=-1) + (self.Yi * qlm).sum(dim=-2)
        ql = (ql * self.coeff).sqrt()
        return ql


def test():
    import numpy as np
    from scipy.special import sph_harm

    from theforce.descriptor.sphcart import cart_coord_to_sph

    lmax = 6

    # using Ql
    rc = 6.0
    Q = Ql(lmax, rc)
    xyz = torch.rand(5, 3)
    q1 = Q(xyz)

    # using scipy
    x, y, z = (xyz[:, j].numpy() for j in range(3))
    r, t, p = cart_coord_to_sph(x, y, z)
    q2 = []
    for l in range(lmax + 1):
        ylm = np.array([sph_harm(m, l, p, t) for m in range(-l, l + 1)])
        cut = (1 - r / rc) ** 2
        qlm = (ylm * cut).sum(axis=-1) / cut.sum()
        ql = (qlm * qlm.conj()).sum() * 4 * pi / (2 * l + 1)
        q2.append(np.sqrt(ql.real))
    q2 = torch.as_tensor(q2)

    # test
    test = q1.allclose(q2)
    print(f"Ql consistent with direct calc using scipy: {test}")


if __name__ == "__main__":
    test()
