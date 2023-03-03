# +
import numpy as np
from numpy import pi


class sph_repr:
    def __init__(self, lmax):

        self.lmax = lmax
        self.lmax_p = lmax + 1

        # pre-calculate
        self.Yoo = np.sqrt(1.0 / (4 * pi))
        self.alp_al = 2 * [[]] + [
            np.array(
                [np.sqrt((4.0 * l * l - 1.0) / (l * l - m * m)) for m in range(l - 1)][
                    ::-1
                ]
            )[:, np.newaxis]
            for l in range(2, self.lmax_p)
        ]
        self.alp_bl = 2 * [[]] + [
            np.array(
                [
                    -np.sqrt(((l - 1.0) ** 2 - m * m) / (4 * (l - 1.0) ** 2 - 1))
                    for m in range(l - 1)
                ][::-1]
            )[:, np.newaxis]
            for l in range(2, self.lmax_p)
        ]
        self.alp_cl = [np.sqrt(2.0 * l + 1.0) for l in range(self.lmax_p)]
        self.alp_dl = [[]] + [
            -np.sqrt(1.0 + 1.0 / (2.0 * l)) for l in range(1, self.lmax_p)
        ]

        # indices: for traversing diagonals
        self.I = [[l + k for l in range(lmax - k + 1)] for k in range(lmax + 1)]
        self.J = [[l for l in range(lmax - k + 1)] for k in range(lmax + 1)]

        # l,m tables
        self.l = np.array(
            [
                [l for m in range(l)] + [m for m in range(l, self.lmax + 1)]
                for l in range(self.lmax + 1)
            ]
        )[:, :, np.newaxis]
        self.m = np.empty_like(self.l)
        for m in range(self.lmax + 1):
            self.m[self.I[m], self.J[m]] = m
            self.m[self.J[m], self.I[m]] = m
        self.m2 = self.m**2

        # lower triangle indices
        self.tril_indices = np.tril_indices(self.lmax + 1, k=-1)

        # l,m related coeffs
        self.coef = np.sqrt(
            (self.l - self.m)
            * (self.l + self.m)
            * (2 * self.l + 1)
            / (2 * self.l - 1.0)
        )[1:, 1:]

    def cart_coord_to_trig(self, x, y, z):
        """points along the z-axis are special, origin is double special"""
        rxy_sq = np.atleast_1d(x * x + y * y)
        rxy = np.sqrt(rxy_sq)
        r_sq = rxy_sq + z * z
        r = np.sqrt(r_sq)
        sin_theta = np.nan_to_num(rxy / r)
        cos_theta = z / r
        cos_theta[np.isnan(cos_theta)] = 1.0
        sin_phi = np.nan_to_num(y / rxy)
        cos_phi = x / rxy
        cos_phi[np.isnan(cos_phi)] = 1.0
        return r, sin_theta, cos_theta, sin_phi, cos_phi

    def ylm(self, x, y, z):
        """
        Inputs: x, y, z Cartesian coordinates
        Returns: r, sin_theta, cos_theta, sin_phi, cos_phi, Y
        r: radius, shape is like x
        sin_theta, cos_theta, sin_phi, cos_phi: sin and cos of theta, phi
        Y: spherical harmonics, shape = (lmax+1,lmax+1,*np.shape(x))
        ------------------------------------------------------------------------
        The imaginary componenst are stored in the upper diagonal of array Y.
        l = 0,...,lmax
        m = 0,...,l
        r: real part
        i: imaginary part

        with lmax=3 this arrangement looks like

            0 1 2 3       0 1 2 3        r i i i
        l = 1 1 2 3   m = 1 0 1 2    Y = r r i i
            2 2 2 3       2 1 0 1        r r r i
            3 3 3 3       3 2 1 0        r r r r

        the full harmonic with l, m (m>0): Y[l,l-m] + 1.0j*Y[l-m,l]
                                    (m=0): Y[l,l]
        """
        r, sin_theta, cos_theta, sin_phi, cos_phi = self.cart_coord_to_trig(x, y, z)
        # alp
        Y = np.empty(
            shape=(self.lmax_p, self.lmax_p, *sin_theta.shape), dtype=sin_theta.dtype
        )
        Y[0, 0] = np.full_like(sin_theta, self.Yoo)
        Y[1, 1] = self.alp_cl[1] * cos_theta * Y[0, 0]
        Y[1, 0] = self.alp_dl[1] * sin_theta * Y[0, 0]
        Y[0, 1] = Y[1, 0]
        for l in range(2, self.lmax_p):
            Y[l, 2 : l + 1] = self.alp_al[l] * (
                cos_theta * Y[l - 1, 1:l] + self.alp_bl[l] * Y[l - 2, : l - 1]
            )
            Y[l, 1] = self.alp_cl[l] * cos_theta * Y[l - 1, 0]
            Y[l, 0] = self.alp_dl[l] * sin_theta * Y[l - 1, 0]
            Y[:l, l] = Y[l, :l]
        # ylm
        c = cos_phi
        s = sin_phi
        Y[self.I[1], self.J[1]] *= c
        Y[self.J[1], self.I[1]] *= s
        for m in range(2, self.lmax_p):
            c, s = cos_phi * c - sin_phi * s, sin_phi * c + cos_phi * s
            Y[self.I[m], self.J[m]] *= c
            Y[self.J[m], self.I[m]] *= s
        return r, sin_theta, cos_theta, sin_phi, cos_phi, Y

    def ylm_rl(self, x, y, z):
        """
        Returns: r, sin_theta, cos_theta, sin_phi, cos_phi, Y
        Y: r**l * Y_l^m( \\theta, \\phi )
        ---------------------------------------------------------
        All same as sph_repr.ylm, only with a r^l multiplied
        to spherical harmonics.

        r**l * Y_l^m becomes  (m>0): Y[l,l-m] + 1.0j*Y[l-m,l]
                              (m=0): Y[l,l]
        ---------------------------------------------------------
        Note that at the special point (0,0,0) theta, phi become 0.
        """
        r, sin_theta, cos_theta, sin_phi, cos_phi = self.cart_coord_to_trig(x, y, z)
        # r^l preparation
        r_sin_theta = r * sin_theta
        r_cos_theta = r * cos_theta
        r2 = r * r
        # alp
        Y = np.empty(
            shape=(self.lmax_p, self.lmax_p, *sin_theta.shape), dtype=sin_theta.dtype
        )
        Y[0, 0] = np.full_like(r_sin_theta, self.Yoo)
        Y[1, 1] = self.alp_cl[1] * r_cos_theta * Y[0, 0]
        Y[1, 0] = self.alp_dl[1] * r_sin_theta * Y[0, 0]
        Y[0, 1] = Y[1, 0]
        for l in range(2, self.lmax_p):
            Y[l, 2 : l + 1] = self.alp_al[l] * (
                r_cos_theta * Y[l - 1, 1:l] + r2 * self.alp_bl[l] * Y[l - 2, : l - 1]
            )
            Y[l, 1] = self.alp_cl[l] * r_cos_theta * Y[l - 1, 0]
            Y[l, 0] = self.alp_dl[l] * r_sin_theta * Y[l - 1, 0]
            Y[:l, l] = Y[l, :l]
        # ylm
        c = cos_phi
        s = sin_phi
        Y[self.I[1], self.J[1]] *= c
        Y[self.J[1], self.I[1]] *= s
        for m in range(2, self.lmax_p):
            c, s = cos_phi * c - sin_phi * s, sin_phi * c + cos_phi * s
            Y[self.I[m], self.J[m]] *= c
            Y[self.J[m], self.I[m]] *= s
        return r, sin_theta, cos_theta, sin_phi, cos_phi, Y

    def ylm_partials(self, sin_theta, cos_theta, Y, with_r=None):
        """
        Return: Y_theta, Y_phi
        i.e. partial derivatives of spherical harmonics wrt theta, phi
        with_r:
        if r^l is multiplied to spherical harmonics (see sph_repr.ylm_rl),
        then "with_r = r" is required for the correct evaluation of
        the partial deriavitives.
        --------------------------------------------------------------
        see ylm for the array storage convention
        """
        # paritial theta
        cot_theta = cos_theta / sin_theta
        Y_theta = cot_theta * self.l * Y
        if with_r is None:
            Y_theta[1:, 1:] -= Y[:-1, :-1] * self.coef / sin_theta
        else:
            Y_theta[1:, 1:] -= with_r * Y[:-1, :-1] * self.coef / sin_theta
        # partial phi
        axes = list(range(len(Y.shape)))
        axes[0], axes[1] = 1, 0
        Y_phi = np.transpose(Y, axes=axes).copy()
        Y_phi[self.tril_indices] *= -1
        Y_phi *= self.m
        return np.nan_to_num(Y_theta), np.nan_to_num(Y_phi)

    def ylm_hessian(self, sin_theta, cos_theta, Y, Y_theta, Y_phi, with_r=None):
        """
        Return: Y_theta_2, Y_phi_2, Y_theta_phi
        i.e. second order partial derivatives of spherical harmonics wrt theta, phi
        with_r:
        if r^l is multiplied to spherical harmonics (see sph_repr.ylm_rl),
        then "with_r = r" is required for the correct evaluation of
        the partial deriavitives.
        --------------------------------------------------------------
        see ylm for the array storage convention
        """
        # paritial theta
        cot_theta = cos_theta / sin_theta
        d_cot_theta = -(1.0 + cot_theta * cot_theta)
        d_i_sin_theta = -cot_theta / sin_theta
        # second order wrt theta
        Y_theta_2 = (d_cot_theta * Y + cot_theta * Y_theta) * self.l
        if with_r is None:
            Y_theta_2[1:, 1:] -= (
                Y_theta[:-1, :-1] / sin_theta + Y[:-1, :-1] * d_i_sin_theta
            ) * self.coef
        else:
            Y_theta_2[1:, 1:] -= (
                with_r
                * (Y_theta[:-1, :-1] / sin_theta + Y[:-1, :-1] * d_i_sin_theta)
                * self.coef
            )
        # second order wrt phi
        Y_phi_2 = -Y * self.m2
        # wrt theta wrt phi
        axes = list(range(len(Y.shape)))
        axes[0], axes[1] = 1, 0
        Y_theta_phi = np.transpose(Y_theta, axes=axes).copy()
        Y_theta_phi[self.tril_indices] *= -1
        Y_theta_phi *= self.m
        # TODO: see if nan_to_num is appropriate here (not checked yet)
        return (np.nan_to_num(a) for a in (Y_theta_2, Y_phi_2, Y_theta_phi))


# test routines ----------------------------------------------------------
def test_sph_repr(n=1000):
    from scipy.special import sph_harm

    from theforce.descriptor.sphcart import cart_coord_to_sph

    lmax = 8
    sph = sph_repr(lmax)
    x = np.concatenate((np.random.uniform(-1.0, 1.0, size=n), [1, 0, 0, 0]))
    y = np.concatenate((np.random.uniform(-1.0, 1.0, size=n), [0, 1, 0, 0]))
    z = np.concatenate((np.random.uniform(-1.0, 1.0, size=n), [0, 0, 1, 0]))
    r, theta, phi = cart_coord_to_sph(x, y, z)
    r, st, ct, sp, cp, Y = sph.ylm(x, y, z)
    r, _, _, _, _, Y_rl = sph.ylm_rl(x, y, z)
    Y_theta, Y_phi = sph.ylm_partials(st, ct, Y)
    Y_theta_rl, Y_phi_rl = sph.ylm_partials(st, ct, Y_rl, with_r=r)
    cott = np.nan_to_num(ct / st)
    errors = []
    for l in range(lmax + 1):
        # m = 0
        rl = r**l
        tmp = sph_harm(0, l, phi, theta)
        errors += [Y[l, l] - tmp, Y_rl[l, l] - rl * tmp]
        if l > 0:
            tmp = (
                np.sqrt(l * (l + 1.0))
                * np.exp(-1.0j * phi)
                * sph_harm(1, l, phi, theta)
            )
            errors += [Y_theta[l, l] - tmp]
            errors += [Y_theta_rl[l, l] - rl * tmp]
        # m > 0
        for m in range(1, l + 1):
            tmp = sph_harm(m, l, phi, theta)
            errors += [
                Y[l, l - m] + 1.0j * Y[l - m, l] - tmp,
                Y_rl[l, l - m] + 1.0j * Y_rl[l - m, l] - rl * tmp,
            ]
            # partial wrt phi
            errors += [Y_phi[l, l - m] + 1.0j * Y_phi[l - m, l] - 1.0j * m * tmp]
            errors += [
                Y_phi_rl[l, l - m] + 1.0j * Y_phi_rl[l - m, l] - 1.0j * rl * m * tmp
            ]
            # partial wrt theta
            tmp2 = np.nan_to_num(m * cott * tmp)
            if m < l:
                tmp2 += (
                    np.sqrt((l - m) * (l + m + 1.0))
                    * np.exp(-1.0j * phi)
                    * sph_harm(m + 1, l, phi, theta)
                )
            errors += [Y_theta[l, l - m] + 1.0j * Y_theta[l - m, l] - tmp2]
            errors += [Y_theta_rl[l, l - m] + 1.0j * Y_theta_rl[l - m, l] - rl * tmp2]

    errors = abs(np.array(errors).reshape(-1))
    test_result = np.allclose(errors, 0.0)
    print(
        """
    comparison with scipy.sph_harm:
    tests included: ylm, ylm_rl, ylm_partials (with_r= None and r)
    all diffs close to zero: {}
    max difference: {}
    """.format(
            test_result, errors.max()
        )
    )
    return test_result


def test_hessian_ylm(lmax=4, N=3):
    from sympy import Derivative, Ynm, symbols

    from theforce.descriptor.sphcart import cart_coord_to_sph

    r_s, theta_s, phi_s = symbols("r theta phi")
    l_s = symbols("l", integer=True, nonnegative=True)
    m_s = symbols("m", integer=True)
    f = Ynm(l_s, m_s, theta_s, phi_s) * r_s**l_s
    # symbolic derivatives
    wrt_theta = Derivative(f, theta_s, 2).doit()
    wrt_cross = Derivative(f, theta_s, phi_s).doit()
    wrt_phi = Derivative(f, phi_s, 2).doit()
    #
    sph = sph_repr(lmax)
    zeros = []
    # random x,y,z
    for _ in range(N):
        x, y, z = np.random.uniform(-1.0, 1.0, size=3)
        r, theta, phi = cart_coord_to_sph(x, y, z)
        subs = {r_s: r, theta_s: theta, phi_s: phi}
        # numeric derivatives
        r, sin_theta, cos_theta, sin_phi, cos_phi, Y = sph.ylm_rl(x, y, z)
        Y_theta, Y_phi = sph.ylm_partials(sin_theta, cos_theta, Y, with_r=r)
        Y_theta_2, Y_phi_2, Y_cross = sph.ylm_hessian(
            sin_theta, cos_theta, Y, Y_theta, Y_phi, with_r=r
        )

        def get(Y, l, m):
            return Y[l, l] if m == 0 else Y[l, l - m] + 1.0j * Y[l - m, l]

        for l in range(lmax + 1):
            subs[l_s] = l
            for m in range(l + 1):
                subs[m_s] = m
                zeros.append(
                    get(Y_theta_2, l, m) - complex(wrt_theta.subs(subs).evalf())
                )
                zeros.append(get(Y_phi_2, l, m) - complex(wrt_phi.subs(subs).evalf()))
                zeros.append(get(Y_cross, l, m) - complex(wrt_cross.subs(subs).evalf()))
    zeros = np.array(zeros)
    test_result = np.allclose(zeros, 0.0)
    maxdiff = max([abs(zeros.min()), abs(zeros.max())])
    print(
        """
    hessian of ylm (through sympy) eqv to sph_repr.ylm_hessian = {}
    maxdiff approx: {}\n""".format(
            test_result, maxdiff
        )
    )
    return test_result


def test_special():
    from scipy.special import sph_harm

    lmax = 2
    sph = sph_repr(lmax)
    r, sin_theta, cos_theta, sin_phi, cos_phi, Y = sph.ylm(0.0, 0.0, 0.0)
    phi, theta = 0, 0
    Y_scipy = np.array([sph_harm(0, l, phi, theta).real for l in range(lmax + 1)])
    print("test at xyz=[0,0,0]: {}".format(np.allclose(Y_scipy, Y.diagonal())))

    # test r^l Ylm, and partials
    r, sin_theta, cos_theta, sin_phi, cos_phi, Y = sph.ylm_rl(0, 0, 0)
    Y_theta, Y_phi = sph.ylm_partials(sin_theta, cos_theta, Y, with_r=r)

    # print(Y_phi[..., 0])
    # print(Y_theta[..., 0])


if __name__ == "__main__":
    test_sph_repr()
    test_hessian_ylm()
    test_special()
