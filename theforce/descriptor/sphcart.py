# +
import numpy as np

tiny = 1.0e-100


# --------------------------------------------------- parsing
def parse_xyz(xyz, order=0):
    """
    order:  order of elements in memory (invariat wrt any reshaping)
            0 for xyz, xyz, xyz, xyz, ...
            1 for xxx..., yyy..., zzz...
    """
    if order == 0:
        a = np.asarray(xyz).reshape(-1, 3)
    elif order == 1:
        a = np.asarray(xyz).reshape(3, -1).T
    return (b[:, 0] for b in np.hsplit(a, 3))


# ---------------------------------------------------- conversions
def cart_coord_to_sph(x, y, z):
    rxy_sq = np.atleast_1d(x * x + y * y)
    r = np.sqrt(rxy_sq + z * z)
    theta = np.arctan2(np.sqrt(rxy_sq), z)
    phi = np.arctan2(y, x)
    return r, theta, phi


def sph_coord_to_cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cart_vec_to_sph(x, y, z, Fx, Fy, Fz):
    rxy_sq = np.atleast_1d(x * x + y * y)
    rxy = np.sqrt(rxy_sq) + tiny
    r_sq = rxy_sq + z * z
    r = np.sqrt(r_sq)
    F_r = (x * Fx + y * Fy + z * Fz) / r
    F_theta = ((x * Fx + y * Fy) * z - rxy_sq * Fz) / (r * rxy)
    F_phi = (-y * Fx + x * Fy) / rxy
    return F_r, F_theta, F_phi


def sph_vec_to_cart(sin_theta, cos_theta, sin_phi, cos_phi, F_r, F_theta, F_phi):
    F_x = sin_theta * cos_phi * F_r + cos_theta * cos_phi * F_theta - sin_phi * F_phi
    F_y = sin_theta * sin_phi * F_r + cos_theta * sin_phi * F_theta + cos_phi * F_phi
    F_z = cos_theta * F_r - sin_theta * F_theta
    return F_x, F_y, F_z


def cart_coord_to_trig(x, y, z):
    rxy_sq = np.atleast_1d(x * x + y * y)
    rxy = np.sqrt(rxy_sq) + tiny
    r_sq = rxy_sq + z * z
    r = np.sqrt(r_sq)
    sin_theta = rxy / r
    cos_theta = z / r
    sin_phi = y / rxy
    cos_phi = x / rxy
    return r, sin_theta, cos_theta, sin_phi, cos_phi


def angles_to_trig(theta, phi):
    return np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)


# ------------------------------------------------------- rotations
def rotation(axis, theta):
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def euler_rotation(alpha, beta, gamma):
    yhat = np.asarray([0, 1.0, 0])
    zhat = np.asarray([0, 0, 1.0])
    R1 = rotation(zhat, gamma)
    R2 = rotation(yhat, beta)
    R3 = rotation(zhat, alpha)
    return np.dot(R3, np.dot(R2, R1))


def rotate(a, b, c, axis, beta, spherical=False):
    """a,b,c =  x,y,z   or   r, theta, phi"""
    if spherical:
        x, y, z = sph_coord_to_cart(a, b, c)
    else:
        x, y, z = a, b, c
    rmat = rotation(axis, beta)
    # c = np.matmul( rmat, np.asarray([x,y,z]) )
    c = [
        rmat[0, 0] * x + rmat[0, 1] * y + rmat[0, 2] * z,
        rmat[1, 0] * x + rmat[1, 1] * y + rmat[1, 2] * z,
        rmat[2, 0] * x + rmat[2, 1] * y + rmat[2, 2] * z,
    ]
    if spherical:
        return cart_coord_to_sph(c[0], c[1], c[2])
    else:
        return c[0], c[1], c[2]


# ----------------------------------------------------------- tests
def rand_sph_coord(N, R=1.0):
    r = np.random.uniform(0, R, size=N)
    t = np.random.uniform(0, np.pi, size=N)
    p = np.random.uniform(0, 2 * np.pi, size=N)
    return r, t, p


def rand_cart_coord(N, R=1.0):
    r, t, p = rand_sph_coord(N, R)
    x, y, z = sph_coord_to_cart(r, t, p)
    return x, y, z


def test_coord_transforms(N=1000):
    r, t, p = rand_sph_coord(N)
    x, y, z = sph_coord_to_cart(r, t, p)
    r2, t2, p2 = cart_coord_to_sph(x, y, z)
    p2 = p2 % (2 * np.pi)
    test1 = np.allclose([r - r2, t - t2, p - p2], 0.0)
    x, y, z = rand_cart_coord(N)
    r, t, p = cart_coord_to_sph(x, y, z)
    x2, y2, z2 = sph_coord_to_cart(r, t, p)
    test2 = np.allclose([x - x2, y - y2, z - z2], 0.0)
    if test1 and test2:
        print("trans coords: test passed")
    else:
        print("trans coords: test failed")


def test_vec_transforms(N=1000):
    r, t, p = rand_sph_coord(N)
    x, y, z = sph_coord_to_cart(r, t, p)
    Fx, Fy, Fz = rand_cart_coord(N)
    Fr, Ft, Fp = cart_vec_to_sph(x, y, z, Fx, Fy, Fz)
    trig = angles_to_trig(t, p)
    _x, _y, _z = sph_vec_to_cart(*trig, Fr, Ft, Fp)
    test = np.allclose([Fx - _x, Fy - _y, Fz - _z], 0.0)
    if test:
        print("trans vec: test passed")
    else:
        print("trans vec:test failed")


def test_rotate(N=1000):
    for _ in range(N):
        axis = np.random.uniform(size=3)
        theta1 = np.random.uniform(0, 2 * np.pi)
        theta2 = np.random.uniform(0, 2 * np.pi)
        theta = theta1 + theta2
        r, t, p = rand_sph_coord(17, R=10.0)
        r1, t1, p1 = rotate(r, t, p, axis, theta1, spherical=True)
        r2, t2, p2 = rotate(r1, t1, p1, axis, theta2, spherical=True)
        r3, t3, p3 = rotate(r, t, p, axis, theta, spherical=True)
        test = np.allclose([r3 - r2, t3 - t2, p3 - p2], 0.0)
        if not test:
            break
    if test:
        print("rot test passed")
    else:
        print("rot test failed")


if __name__ == "__main__":

    test_coord_transforms()

    test_vec_transforms()

    test_rotate()
