# +
from math import pi as math_pi

import torch
from torch.nn.modules import Module

pi = torch.tensor(math_pi)


def split_and_rotate_tiny_if_too_close_to_zaxis(xyz, tiny_angle=1e-2):
    """
    tiny_angle is chosen such that the calculated gradients (in Ylm) are
    consistent with torch.autograd.
    This depends both on precision and scale of xyz.
    """
    tol = tiny_angle * xyz[:, 2].abs()
    if ((xyz[:, 0].abs() < tol) & (xyz[:, 1].abs() < tol)).any():
        x = xyz[:, 0]
        y = xyz[:, 1] - tiny_angle * xyz[:, 2]
        z = tiny_angle * xyz[:, 1] + xyz[:, 2]
        return x, y, z, tiny_angle
    else:
        return xyz[:, 0], xyz[:, 1], xyz[:, 2], 0


def cart_coord_to_trig(x, y, z):
    rxy_sq = x * x + y * y
    rxy = rxy_sq.sqrt()
    r = (rxy_sq + z * z).sqrt()
    sin_theta = rxy / r
    cos_theta = z / r
    sin_phi = y / rxy
    cos_phi = x / rxy
    return r, sin_theta, cos_theta, sin_phi, cos_phi


def sph_vec_to_cart(sin_theta, cos_theta, sin_phi, cos_phi, F_r, F_theta, F_phi):
    F_x = sin_theta * cos_phi * F_r + cos_theta * cos_phi * F_theta - sin_phi * F_phi
    F_y = sin_theta * sin_phi * F_r + cos_theta * sin_phi * F_theta + cos_phi * F_phi
    F_z = cos_theta * F_r - sin_theta * F_theta
    return F_x, F_y, F_z


class Ylm(Module):
    """
    Calculates spherical harmonics from the Cartesian coordinates,
    using pytorch.
    """

    def __init__(self, lmax):
        super().__init__()

        self.lmax = lmax

        # pre-calculate
        self.Yoo = torch.sqrt(1.0 / (4 * pi))
        self.alp_al = 2 * [[]] + [
            torch.tensor(
                [
                    torch.tensor((4.0 * l * l - 1.0) / (l * l - m * m)).sqrt()
                    for m in range(l - 1)
                ]
            )[:, None]
            for l in range(2, lmax + 1)
        ]
        self.alp_bl = 2 * [[]] + [
            torch.tensor(
                [
                    -torch.tensor(
                        ((l - 1.0) ** 2 - m * m) / (4 * (l - 1.0) ** 2 - 1)
                    ).sqrt()
                    for m in range(l - 1)
                ]
            )[:, None]
            for l in range(2, lmax + 1)
        ]
        self.alp_cl = [torch.tensor(2.0 * l + 1.0).sqrt() for l in range(lmax + 1)]
        self.alp_dl = [[]] + [
            -torch.tensor(1.0 + 1.0 / (2.0 * l)).sqrt() for l in range(1, lmax + 1)
        ]

        # indices: for traversing diagonals
        self.I = [[l + k for l in range(lmax - k + 1)] for k in range(lmax + 1)]
        self.J = [[l for l in range(lmax - k + 1)] for k in range(lmax + 1)]

        # l,m tables
        self.l = torch.tensor(
            [
                [l for m in range(l)] + [m for m in range(l, self.lmax + 1)]
                for l in range(self.lmax + 1)
            ]
        )[:, :, None]
        self.m = torch.zeros_like(self.l)
        for m in range(self.lmax + 1):
            self.m[self.I[m], self.J[m]] = m
            self.m[self.J[m], self.I[m]] = m

        # lower triangle indices
        one = torch.ones(lmax + 1, lmax + 1)
        self.sign = (-torch.tril(one, diagonal=-1) + torch.triu(one))[..., None]

        # l,m related coeffs
        self.coef = (
            ((self.l - self.m) * (self.l + self.m) * (2 * self.l + 1)).float()
            / (2 * self.l - 1).float()
        )[1:, 1:].sqrt()

        # floats
        self.l_float = self.l.type(one.type())
        self.m_float = self.m.type(one.type())
        self.coef = self.coef.type(one.type())

    def forward(self, xyz, with_r=None, grad=True, spherical_grads=False):
        """
        Calculates Y = r**l * Ylm(theta, phi).
        To eliminate r**l factor set with_r=1.
        If grad=True, gradient of Y wrt xyz will be calculated.
        The imaginary componenst are stored in the upper diagonal of array Y.
        l = 0, ..., lmax
        m = 0, ..., l
        r: real part
        i: imaginary part

        with lmax=3 this arrangement looks like

            0 1 2 3       0 1 2 3        r i i i
        l = 1 1 2 3   m = 1 0 1 2    Y = r r i i
            2 2 2 3       2 1 0 1        r r r i
            3 3 3 3       3 2 1 0        r r r r

        the full harmonic with l, m (m>0): Y[l,l-m] + 1.0j*Y[l-m,l]
                                    (m=0): Y[l,l]
        ------------------------------------------------------------
        Note: if one of vectors becomes too close to z_axis, the entire system
        will be rotated slightly, and in turn, the calculated gradients will be
        rotated by an inverse rotation.
        This way the returned gradients are consistent with torch.autograd.
        """
        x, y, z, angle = split_and_rotate_tiny_if_too_close_to_zaxis(xyz)
        r, sin_theta, cos_theta, sin_phi, cos_phi = cart_coord_to_trig(x, y, z)
        _r = r if with_r is None else torch.as_tensor(with_r).type(xyz.type())
        r2 = _r * _r
        r_sin_theta = _r * sin_theta
        r_cos_theta = _r * cos_theta
        # Associated Legendre polynomials
        alp = [[torch.full_like(sin_theta, self.Yoo)]]
        for l in range(1, self.lmax + 1):
            alp += [
                [
                    self.alp_al[l][m]
                    * (
                        r_cos_theta * alp[l - 1][m]
                        + r2 * self.alp_bl[l][m] * alp[l - 2][m]
                    )
                    for m in range(l - 1)
                ]
                + [self.alp_cl[l] * r_cos_theta * alp[l - 1][l - 1]]
                + [self.alp_dl[l] * r_sin_theta * alp[l - 1][l - 1]]
            ]
        # sin, cos of m*phi
        sin = [torch.zeros_like(sin_phi), sin_phi]
        cos = [torch.ones_like(cos_phi), cos_phi]
        for m in range(2, self.lmax + 1):
            s = sin_phi * cos[-1] + cos_phi * sin[-1]
            c = cos_phi * cos[-1] - sin_phi * sin[-1]
            sin += [s]
            cos += [c]
        # Spherical Harmonics
        n = sin_theta.size(0)
        Yr = torch.cat(
            [
                torch.cat(
                    [(alp[l][m] * cos[m]).view(1, n) for m in range(l, -1, -1)]
                    + [torch.zeros(self.lmax - l, n)]
                ).view(self.lmax + 1, 1, n)
                for l in range(self.lmax + 1)
            ],
            dim=1,
        ).permute(1, 0, 2)
        Yi = torch.cat(
            [
                torch.cat(
                    [(alp[l][m] * sin[m]).view(1, n) for m in range(l, -1, -1)]
                    + [torch.zeros(self.lmax - l, n)]
                ).view(self.lmax + 1, 1, n)
                for l in range(self.lmax + 1)
            ],
            dim=1,
        )
        Y = Yr + Yi
        # partial derivatives
        if grad:
            if with_r is None:
                Y_r = self.l_float * Y / r
            else:
                Y_r = torch.zeros_like(Y)
            Y_theta = cos_theta * self.l_float * Y / sin_theta
            Y_theta[1:, 1:] -= _r * Y[:-1, :-1] * self.coef / sin_theta
            Y_phi = Y.clone().permute(1, 0, 2) * self.sign * self.m_float
            if spherical_grads:
                return Y, torch.stack([Y_r, Y_theta, Y_phi], dim=-1)
            else:
                cart = sph_vec_to_cart(
                    sin_theta,
                    cos_theta,
                    sin_phi,
                    cos_phi,
                    Y_r,
                    Y_theta / r,
                    Y_phi / (r * sin_theta),
                )
                if angle > 0:
                    dY = torch.stack(
                        [
                            cart[0],
                            cart[1] + angle * cart[2],
                            -angle * cart[1] + cart[2],
                        ],
                        dim=-1,
                    )
                else:
                    dY = torch.stack(cart, dim=-1)
                return Y, dY
        else:
            return Y


def compare_with_numpy_version():
    import numpy as np

    from theforce.descriptor.sph_repr import sph_repr

    # torch.set_default_dtype(torch.double)
    # torch.set_default_tensor_type(torch.DoubleTensor)

    lmax = 4
    sph_np = sph_repr(lmax)
    sph_torch = Ylm(lmax)

    n = 7
    xyz = torch.rand(n, 3, requires_grad=True)
    x, y, z = (a.detach().numpy().reshape(-1) for a in xyz.split(1, dim=-1))

    # test without r^l
    r, sin_theta, cos_theta, sin_phi, cos_phi, Y_np = sph_np.ylm(x, y, z)
    Y_theta_np, Y_phi_np = sph_np.ylm_partials(sin_theta, cos_theta, Y_np, with_r=None)
    Y, grads = sph_torch(xyz, with_r=1, spherical_grads=True)
    Y_r, Y_theta, Y_phi = grads[..., 0], grads[..., 1], grads[..., 2]
    a = Y.allclose(torch.tensor(Y_np).type(xyz.type()))
    b = Y_theta.allclose(torch.tensor(Y_theta_np).type(xyz.type()))
    c = Y_phi.allclose(torch.tensor(Y_phi_np).type(xyz.type()))
    ea = (Y - torch.tensor(Y_np).type(xyz.type())).abs().max().data
    eb = (Y_theta - torch.tensor(Y_theta_np).type(xyz.type())).abs().max().data
    ec = (Y_phi - torch.tensor(Y_phi_np).type(xyz.type())).abs().max().data
    print(a, b, c, ea, eb, ec)

    # test with r^l
    r, sin_theta, cos_theta, sin_phi, cos_phi, Y_np = sph_np.ylm_rl(x, y, z)
    Y_theta_np, Y_phi_np = sph_np.ylm_partials(sin_theta, cos_theta, Y_np, with_r=r)
    Y, grads = sph_torch(xyz, with_r=None, spherical_grads=True)
    Y_r, Y_theta, Y_phi = grads[..., 0], grads[..., 1], grads[..., 2]
    a = Y.allclose(torch.tensor(Y_np).type(xyz.type()))
    b = Y_theta.allclose(torch.tensor(Y_theta_np).type(xyz.type()))
    c = Y_phi.allclose(torch.tensor(Y_phi_np).type(xyz.type()))
    ea = (Y - torch.tensor(Y_np).type(xyz.type())).abs().max().data
    eb = (Y_theta - torch.tensor(Y_theta_np).type(xyz.type())).abs().max().data
    ec = (Y_phi - torch.tensor(Y_phi_np).type(xyz.type())).abs().max().data
    print(a, b, c, ea, eb, ec)

    Y.sum().backward(retain_graph=True)
    Y_theta.sum().backward(retain_graph=True)
    Y_phi.sum().backward()
    xyz.grad


def compare_grads_with_autograd():
    torch.set_default_tensor_type(torch.DoubleTensor)

    sph = Ylm(3)
    xyz = torch.rand(10, 3, requires_grad=True)
    Y, grad = sph(xyz, grad=True, with_r=1)
    Y.sum().backward()
    a = xyz.grad.allclose(grad.sum(dim=(0, 1)))
    ea = (xyz.grad - grad.sum(dim=(0, 1))).abs().max().data
    print(a, ea)

    xyz = torch.rand(10, 3, requires_grad=True)
    Y, grad = sph(xyz, grad=True, with_r=None)
    Y.sum().backward()
    a = xyz.grad.allclose(grad.sum(dim=(0, 1)))
    ea = (xyz.grad - grad.sum(dim=(0, 1))).abs().max().data
    print(a, ea)

    xyz = torch.tensor([[0, 0, 1.0]], requires_grad=True)
    Y, grad = sph(xyz, grad=True, with_r=None)
    Y.sum().backward()
    a = xyz.grad.allclose(grad.sum(dim=(0, 1)))
    ea = (xyz.grad - grad.sum(dim=(0, 1))).abs().max().data
    print(a, ea)


if __name__ == "__main__":
    compare_with_numpy_version()
    compare_grads_with_autograd()
