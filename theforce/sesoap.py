
# coding: utf-8

# In[ ]:


from math import factorial as fac
import numpy as np
from theforce.sph_repr import sph_repr


class sesoap:

    def __init__(self, lmax, nmax, radial, modify_scale=None):
        """
        lmax: maximum l in r^l * Ylm terms (Ylm are spherical harmonics)
        nmax: maximum n in r^(2n) terms
        radial: radial function e.g. quadratic_cutoff, gaussian, etc.
        modify_scale: for fine-tuning the optimal length scale (see below)
        --------------------------------------------------------------------
        An optimal length scale is calculated which depends on lmax, nmax,
        radial.rc, and an empirical factor (1.333).
        If a positive number is passed as modify_scale, it will multiply
        the calculated length scale by this number.
        Thus a float close to 1 should do the trick.
        If a negative number is passed as modify_scale, then its abs value
        will be passed directly as the scale.
        """
        self.lmax = lmax
        self.nmax = nmax
        self.radial = radial

        self.sph = sph_repr(lmax)

        # prepare some stuff
        self._m = [([l for m in range(0, l+1)] + [m for m in range(0, l)],
                    [m for m in range(0, l+1)] + [l for m in range(0, l)])
                   for l in range(0, lmax+1)]

        # lower triangle indices
        self.In, self.Jn = np.tril_indices(nmax+1)

        # l,n,n'-dependent constant
        a_ln = np.array([[1. / ((2*l+1) * 2**(2*n+l) * fac(n) * fac(n+l))
                          for n in range(nmax+1)] for l in range(lmax+1)])
        tmp = self.compress(
            np.sqrt(np.einsum('ln,lm->lnm', a_ln, a_ln)), 'lnn')
        self.lnnp_c = [tmp, tmp[:, np.newaxis], tmp[:, np.newaxis, np.newaxis]]

        # dim of descriptor after compression
        self.dim = (lmax+1)*((nmax+1)*(nmax+2)//2)

        # prepare for broadcasting
        self.rns = 2 * np.arange(self.nmax+1).reshape(nmax+1, 1, 1, 1)

        # l, n, np compressed
        _ar = np.arange(max(lmax, nmax)+1)
        byte = _ar.strides[0]
        self._il = self.compress(np.lib.stride_tricks.as_strided(
            _ar, shape=(lmax+1, nmax+1, nmax+1), strides=(byte, 0, 0)), 'lnn')
        self._in = self.compress(np.lib.stride_tricks.as_strided(
            _ar, shape=(lmax+1, nmax+1, nmax+1), strides=(0, byte, 0)), 'lnn')
        self._inp = self.compress(np.lib.stride_tricks.as_strided(
            _ar, shape=(lmax+1, nmax+1, nmax+1), strides=(0, 0, byte)), 'lnn')

        # configuring the length scale
        self._empirical = 1.333
        self._opt_rc = self.__opt_rc__(1.0) * self._empirical
        scale = self.radial.rc / self._opt_rc
        if modify_scale is None:
            self.scale = scale
        elif modify_scale > 0:
            self.scale = scale * modify_scale
        else:
            self.scale = abs(modify_scale)
        self.radial.rc /= self.scale

        # input preprocessing function
        self.scaling = lambda r: r / self.scale

    def parse_xyz(self, xyz, order=0):
        """
        order:  order of elements in memory -> invariat wrt any reshaping
                0 for xyz, xyz, xyz, xyz, ...
                1 for xxx..., yyy..., zzz...
        ------------------------------------------------------------------
        it will transform to 0-order before preprocessing inputs
        """
        if order == 0:
            a = self.scaling(np.asarray(xyz).reshape(-1, 3))
        elif order == 1:
            a = self.scaling(np.asarray(xyz).reshape(3, -1).T)
        return (b[:, 0] for b in np.hsplit(a, 3))

    def descriptor(self, xyz, order=0, normalize=True):
        """
        Inputs:   xyz -> Cartesian coordinates, see parse_xyz
        order:    0 or 1, see parse_xyz
        Returns:  p -> compressed (1d) descriptor
        """
        r, _, _, _, _, Y = self.sph.ylm_rl(*self.parse_xyz(xyz, order=order))
        R, _ = self.radial.radial(r)
        s = (R * r**self.rns * Y).sum(axis=-1)
        p = self.soap_dot(s, s, reverse=False)
        if normalize:
            norm = np.linalg.norm(p)
            if norm > 0.0:
                p /= norm
        return p

    def derivatives(self, xyz, order=0, normalize=True, sumj=True, cart=True, flatten=False):
        """
        Inputs:   xyz -> Cartesian coordinates, see parse_xyz
        order:    0 or 1, see parse_xyz
        Returns:  p, q
        sumj:     perform the summation over atoms
        ---------------------------------------
        p:        compressed (1d) descriptor
        q:        [dp_dx, dp_dy, dp_dz]
                  (or gradient in sph coords if cart=False)
                  the order will be consistent with input order
        """
        r, sin_theta, cos_theta, sin_phi, cos_phi, Y = self.sph.ylm_rl(
            *self.parse_xyz(xyz, order=order))
        Y_theta, Y_phi = self.sph.ylm_partials(
            sin_theta, cos_theta, Y, with_r=r)
        R, dR = self.radial.radial(r)
        rns = r**self.rns
        R_rns = R * rns
        # descriptor
        s = (R_rns * Y).sum(axis=-1)
        p = self.soap_dot(s, s, reverse=False)
        # gradients
        rns_plus_l = self.rns + self.sph.l
        Y_theta /= r
        Y_phi /= r*sin_theta
        q1 = self.soap_dot(
            s, ((dR * rns + R_rns * rns_plus_l / r) * Y), jb='j')
        q2 = self.soap_dot(s, (R_rns * Y_theta), jb='j')
        q3 = self.soap_dot(s, (R_rns * Y_phi), jb='j')
        if cart:
            q1, q2, q3 = self.sph_vec_to_cart(
                sin_theta, cos_theta, sin_phi, cos_phi, q1, q2, q3)
        q = np.stack([q1, q2, q3], axis=2-order)
        if normalize:
            norm = np.linalg.norm(p)
            if norm > 0.0:
                p /= norm
                q /= norm
                q -= p[:, None, None] * (p[:, None, None] * q).sum(axis=0)
        if sumj:
            return p, self.scaling(q.sum(axis=order+1))
        else:
            if flatten:
                return p, self.scaling(q).reshape(self.dim, -1)
            else:
                return p, self.scaling(q)

    # ------------------- convenience functions -------------------------------------------------
    @staticmethod
    def sph_vec_to_cart(sin_theta, cos_theta, sin_phi, cos_phi, F_r, F_theta, F_phi):
        F_x = sin_theta * cos_phi * F_r + cos_theta * cos_phi * F_theta - sin_phi * F_phi
        F_y = sin_theta * sin_phi * F_r + cos_theta * sin_phi * F_theta + cos_phi * F_phi
        F_z = cos_theta * F_r - sin_theta * F_theta
        return F_x, F_y, F_z

    def soap_dot(self, a, b, ja='', jb='', reverse=True):
        jc = ja
        if ja == jb:
            jd = ja
            jr = ja
        elif ja != jb:
            jd = ja+jb
            jr = jb+ja
        c = np.einsum('n...'+ja+',m...'+jb+'->nm...'+jd, a, b)
        if reverse:
            c += np.einsum('n...'+jb+',m...'+ja+'->nm...'+jr, b, a)
        return self.compress(self.sum_all_m(c), 'lnn'+jd) * self.lnnp_c[len(jd)]

    def sum_all_m(self, c):
        rank = len(c.shape)
        if rank == 6:
            return np.array([2*c[:, :, self._m[l][0], self._m[l][1], :, :].sum(axis=-3)
                             - c[:, :, l, l, :, :] for l in range(self.lmax+1)])
        elif rank == 5:
            return np.array([2*c[:, :, self._m[l][0], self._m[l][1], :].sum(axis=-2)
                             - c[:, :, l, l, :] for l in range(self.lmax+1)])
        elif rank == 4:
            return np.array([2*c[:, :, self._m[l][0], self._m[l][1]].sum(axis=-1)
                             - c[:, :, l, l] for l in range(self.lmax+1)])

    def compress(self, a, type):
        if type == 'lnn':
            return a[:, self.In, self.Jn].reshape(-1)
        elif type == 'lnnj' or type == 'lnnk':
            j = a.shape[-1]
            return a[:, self.In, self.Jn].reshape(-1, j)
        elif type == '3lnn':
            return a[:, :, self.In, self.Jn].reshape((3, -1))
        elif type == '3lnnj' or type == '3lnnk':
            j = a.shape[-1]
            return a[:, :, self.In, self.Jn, :].reshape((3, -1, j))
        elif type == 'lnnjk' or type == 'lnnkj':
            j = a.shape[-1]
            return a[:, self.In, self.Jn].reshape(-1, j, j)
        else:
            print("type {} not defined yet, for matrix with shape {}".format(
                type, a.shape))

    def decompress(self, v, type, n=None, l=None):
        if n is None:
            n = self.nmax
        if l is None:
            l = self.lmax
        d = len(v.shape)
        if type == 'lnn':
            a = np.empty(shape=(l+1, n+1, n+1), dtype=v.dtype)
            a[:, self.In, self.Jn] = v.reshape((l+1, (n+1)*(n+2)//2))
            a[:, self.Jn, self.In] = a[:, self.In, self.Jn]
        elif type == 'lnnj' or type == 'lnnk':
            j = v.shape[-1]
            a = np.empty(shape=(l+1, n+1, n+1, j), dtype=v.dtype)
            a[:, self.In, self.Jn, :] = v.reshape((l+1, (n+1)*(n+2)//2, j))
            a[:, self.Jn, self.In, :] = a[:, self.In, self.Jn, :]
        elif type == '3lnn':
            a = np.empty(shape=(3, l+1, n+1, n+1), dtype=v.dtype)
            a[:, :, self.In, self.Jn] = v.reshape((3, l+1, (n+1)*(n+2)//2))
            a[:, :, self.Jn, self.In] = a[:, :, self.In, self.Jn]
        elif type == '3lnnj' or type == '3lnnk':
            j = v.shape[-1]
            a = np.empty(shape=(3, l+1, n+1, n+1, j), dtype=v.dtype)
            a[:, :, self.In, self.Jn, :] = v.reshape(
                (3, l+1, (n+1)*(n+2)//2, j))
            a[:, :, self.Jn, self.In, :] = a[:, :, self.In, self.Jn, :]
        elif type == 'lnnjk' or type == 'lnnkj':
            j = v.shape[-1]
            a = np.empty(shape=(l+1, n+1, n+1, j, j), dtype=v.dtype)
            a[:, self.In, self.Jn, :] = v.reshape((l+1, (n+1)*(n+2)//2, j, j))
            a[:, self.Jn, self.In, :] = a[:, self.In, self.Jn, :, :]
        else:
            print("type {} not defined yet, for matrix with shape {}".format(
                type, a.shape))
        return a

# optimizations of length scales -------------------------------------------------------
    @staticmethod
    def __approx_cnlm__(l, n, r_c, rho):
        """ retruns: \int rho * r^(2n+l) * (1-r/rc)^2 * 4 pi r^2 dr """
        c = 4*np.pi*rho
        integ = c*2*r_c**5*r_c**l*r_c**(2*n) / (l**3*r_c**2 + 6*l**2*n*r_c**2 + 12*l**2*r_c**2 +
                                                12*l*n**2*r_c**2 + 48*l*n*r_c**2 + 47*l*r_c**2 +
                                                8*n**3*r_c**2 + 48*n**2*r_c**2 + 94*n*r_c**2 + 60*r_c**2)
        return integ

    def __approx_plnn__(self, l, n, m, r_c, rho):
        """ approx sum over m: cnlm cn'lm """
        return np.sqrt(2*l+1) * self.__approx_cnlm__(l, n, r_c, rho) * self.__approx_cnlm__(l, m, r_c, rho) *             self.lnnp_c[0]

    def __opt_rc__(self, rho):
        from scipy.optimize import minimize

        def objective(rc):
            p = self.__approx_plnn__(
                self._il, self._in, self._inp, rc, rho)
            norm = np.linalg.norm(p)
            p /= norm
            return p.var()
        res = minimize(objective, 0.1, options={'gtol': 1e-10})
        return res.x[0]


# tests ----------------------------------------------------------------------------------
def test_sesoap():
    """ trying to regenerate numbers obtained by symbolic calculations using sympy """
    from theforce.radial_funcs import quadratic_cutoff
    from theforce.sphcart import cart_vec_to_sph, rotate
    x = np.array([0.175, 0.884, -0.87, 0.354, -0.082] + [3.1])  # one outlier
    y = np.array([-0.791, 0.116, 0.19, -0.832, 0.184] + [0.0])
    z = np.array([0.387, 0.761, 0.655, -0.528, 0.973] + [0.0])
    env = sesoap(2, 2, quadratic_cutoff(3.0), modify_scale=-1.)
    p_ = env.descriptor([x, y, z], order=1, normalize=False)
    p_dc, q_dc = env.derivatives(
        [x, y, z], order=1, normalize=False, cart=False)
    q_dc = q_dc.T                                              # order stuff
    p_ = env.decompress(p_, 'lnn')
    p_d = env.decompress(p_dc, 'lnn')
    q_d = env.decompress(q_dc, '3lnn')
    ref_p = [np.array([[[0.36174603, 0.39013356, 0.43448023],
                        [0.39013356, 0.42074877, 0.46857549],
                        [0.43448023, 0.46857549, 0.5218387]],

                       [[0.2906253, 0.30558356, 0.33600938],
                        [0.30558356, 0.3246583, 0.36077952],
                        [0.33600938, 0.36077952, 0.40524778]],

                       [[0.16241845, 0.18307552, 0.20443194],
                        [0.18307552, 0.22340802, 0.26811937],
                        [0.20443194, 0.26811937, 0.34109511]]]),
             np.array([[[-0.73777549, -0.05089412,  0.74691856],
                        [-0.05089412,  0.74833475,  1.70005743],
                        [0.74691856,  1.70005743,  2.85847646]],

                       [[-0.01237519,  0.56690766,  1.23261539],
                        [0.56690766,  1.21157686,  1.99318763],
                        [1.23261539,  1.99318763,  2.95749108]],

                       [[0.27361894,  0.63696076,  1.08095971],
                        [0.63696076,  1.15336381,  1.84451275],
                        [1.08095971,  1.84451275,  2.9120592]]]),
             np.array([[[0.,  0.,  0.],
                        [0.,  0.,  0.],
                        [0.,  0.,  0.]],

                       [[-0.81797727, -0.88483089, -0.99106192],
                        [-0.88483089, -0.95446211, -1.06668809],
                        [-0.99106192, -1.06668809, -1.18983543]],

                       [[0.03152424,  0.0597677,  0.07161054],
                        [0.0597677,  0.11466049,  0.15943685],
                        [0.07161054,  0.15943685,  0.24410156]]]),
             np.array([[[0.,  0.,  0.],
                        [0.,  0.,  0.],
                        [0.,  0.,  0.]],

                       [[0.01059708,  0.00517264, -0.00218289],
                        [0.00517264, -0.00037216, -0.00786604],
                        [-0.00218289, -0.00786604, -0.01549284]],

                       [[0.02103876,  0.00576316, -0.01632531],
                        [0.00576316, -0.01022614, -0.03301236],
                        [-0.01632531, -0.03301236, -0.0564123]]])]
    ref_p *= env.decompress(env.lnnp_c[0], 'lnn')

    print("\nTesting validity of sesoap ...")
    print(np.allclose(p_-ref_p[0], 0.0))
    print(np.allclose(p_d-ref_p[0], 0.0))
    for k in range(3):
        print(np.allclose(q_d[k]-ref_p[k+1], 0.0))

    pj, qj = env.derivatives(
        (x, y, z), order=1, sumj=False, cart=True, normalize=False)
    # order stuff
    qj = np.transpose(qj, axes=(1, 0, 2))
    pj_ = env.decompress(pj, 'lnn')
    qj_ = env.decompress(qj, '3lnnj')
    h = np.array(cart_vec_to_sph(x, y, z, qj_[0], qj_[1], qj_[2]))
    print(np.allclose(h.sum(axis=-1)-ref_p[1:], 0.0))

    pj, qj = env.derivatives(
        (x, y, z), order=1, sumj=True, cart=True, normalize=False)
    qj = qj.T                                                       # order stuff

    pj_ = env.decompress(pj, 'lnn')
    qj__ = env.decompress(qj, '3lnn')
    print(np.allclose(qj_.sum(axis=-1)-qj__, 0.0))

    # test rotations
    axis = np.random.uniform(size=3)
    beta = np.random.uniform(0, 2*np.pi)
    xx, yy, zz = rotate(x, y, z, axis, beta)
    p, q = env.derivatives([x, y, z], order=1, sumj=False,
                           cart=True, normalize=True)
    pp, qq = env.derivatives((xx, yy, zz), order=1, sumj=False,
                             cart=True, normalize=True)
    q, qq = (np.transpose(a, axes=(1, 0, 2))
             for a in [q, qq])          # order stuff
    rot = np.array(rotate(q[0], q[1], q[2], axis, beta))
    print(np.allclose(rot-qq, 0.0))
    print(63*'-'+' done\n')


def test_sesoap_performance(n=30, N=100):
    import time
    from theforce.radial_funcs import quadratic_cutoff
    print("\nTesting speed of sesoap with random xyz[{},3]".format(n))

    # np.random
    start = time.time()
    for _ in range(N):
        x, y, z = (np.random.uniform(-1., 1., size=n) for _ in range(3))
    finish = time.time()
    delta1 = (finish-start)/N
    print("t1: {} Sec per np.random.uniform(shape=({},3))".format(delta1, n))

    env = sesoap(5, 5, quadratic_cutoff(3.0))

    # descriptor
    start = time.time()
    for _ in range(N):
        x, y, z = (np.random.uniform(-1., 1., size=n) for _ in range(3))
        p = env.descriptor((x, y, z), order=1)
    finish = time.time()
    delta2 = (finish-start)/N
    print("t2: {} Sec per descriptor".format(delta2))
    # derivatives
    start = time.time()
    for _ in range(N):
        x, y, z = (np.random.uniform(-1., 1., size=n) for _ in range(3))
        p, q = env.derivatives([x, y, z], order=1)
    finish = time.time()
    delta3 = (finish-start)/N
    print("t3: {} Sec per derivatives (j-reduced)".format(delta3))

    start = time.time()
    for _ in range(N):
        # x, y, z = ( np.random.uniform(-1.,1.,size=n) for _ in range(3) )
        xyz = np.random.uniform(-1., 1., size=(n, 3))
        p, q = env.derivatives(xyz, sumj=False, cart=True)
    finish = time.time()
    delta4 = (finish-start)/N
    print("t4: {} Sec per full derivatives".format(delta4))

    print("performance measure t2/t1: {}".format(delta2/delta1))
    print(63*'-'+' done\n')


def test_derivatives(n=20, rc=1., normalize=True, N=100, atol=1e-10):
    """ 
    Testing derivatives of sesoap by comparison to finite differences.
    For numerical consistency, the second order derivative is also 
    calculated as a measure (propto) of allowable error.
    """
    from theforce.sphcart import rand_cart_coord
    from theforce.radial_funcs import quadratic_cutoff

    env = sesoap(5, 5, quadratic_cutoff(rc))
    delta = 1e-3 * rc
    skin = 10 * delta

    tests, failures, r_failed = [], [], []
    for _ in range(N):

        x, y, z = rand_cart_coord(n, rc-skin)
        p1, q = env.derivatives(
            [x, y, z], order=1, normalize=normalize, sumj=False)

        # choose a particle, skip if r is too small, reduce this threshold to see some errors!
        r = 0
        while r < skin:
            j = np.random.choice(n)
            r = np.sqrt(x[j]**2 + y[j]**2 + z[j]**2)

        # a random direction
        u = np.random.uniform(size=3)
        u /= np.linalg.norm(u)

        # displace j
        x[j] += delta * u[0]
        y[j] += delta * u[1]
        z[j] += delta * u[2]

        # dp/delta via finite differences
        p2 = env.descriptor([x, y, z], order=1, normalize=normalize)
        fd = (p2 - p1) / delta

        # grad in direction of u
        grad = (q[:, :, j] * u).sum(axis=-1)

        # displace one more time to estimate the sec-order diff
        x[j] += delta * u[0]
        y[j] += delta * u[1]
        z[j] += delta * u[2]
        p3 = env.descriptor([x, y, z], order=1, normalize=normalize)
        sec_dr2 = (p1 + p3 - 2 * p2)

        # check: |f(r+dr) - f(r) - grad(f).dr| < |grad^2(f).dr^2|
        diff = abs(fd - grad)
        tol = abs(sec_dr2/delta) * 5 + atol
        ok = diff < tol
        tests += list(ok)
        if not ok.all():
            failures += list(diff[np.logical_not(ok)])
            r_failed += [r]

    num = len(tests)
    passed = sum(tests)
    failed = num - passed
    print('\ntesting derivatives with finite differences:')
    print('passed: {}'.format(passed))
    print('failed: {} \t (={} %)'.format(failed, failed/num))

    if failed > 0:
        print('maximum failure (with atol={}): {}'.format(
            atol, max(failures)))

    print(63*'-'+' done\n')


if __name__ == '__main__':

    test_sesoap()

    test_derivatives()

    test_sesoap_performance()

