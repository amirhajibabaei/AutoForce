# +
import numpy as np
from numpy.linalg import norm
import torch


def generate_random_cluster(distances):
    """
    distances:
        A list of distances with length n,
        generates a cluster with n+1 points.
        The first point is [0, 0, 0].

    The points are selected such that the nearest neighbor
    of each point with index k > 0 is equal to distances[k-1].
    In other words:

        get_distances(cls).min(axis=0)[1:] == distances

    See the test in proceeding.
    """
    cls = np.zeros((1, 3))
    for dist in distances:
        u = random_unit_vector()
        proj = (u*cls).sum(axis=1)
        order = np.argsort(proj)[::-1]
        for j, k in enumerate(order):
            x = proj[k]
            y = norm(cls[k]-x*u)
            if y < dist:
                x += np.sqrt(dist**2-y**2)
                break
        for k in order[j:]:
            if x-proj[k] > dist:
                break
            y = norm(cls[k]-x*u)
            if y < dist:
                z = norm(cls[k]-proj[k]*u)
                x = proj[k] + np.sqrt(dist**2-z**2)
        cls = np.r_[cls, x*u]
    return cls


def random_unit_vector(n=1):
    u = np.random.uniform(-1., 1., size=(n, 3))
    u /= norm(u, axis=1, keepdims=True)
    return u


def get_distances(cls):
    """
    Returns the distance matrix with diagonal elements (=0) eliminated.
    """
    d = np.linalg.norm(cls[None]-cls[:, None], axis=-1)
    return skip_diag_strided(d)


# -----------------------------------------------------------------------------------
# See: https://stackoverflow.com/questions/46736258/deleting-diagonal-elements-of-a-numpy-array
def skip_diag_masking(A):
    return A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)


def skip_diag_broadcasting(A):
    m = A.shape[0]
    idx = (np.arange(1, m+1) + (m+1)*np.arange(m-1)[:, None]).reshape(m, -1)
    return A.ravel()[idx]


def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1, m), strides=(s0+s1, s1)).reshape(m, -1)
# ------------------------------------------------------------------------------------


def test_generate_random_cluster():
    distances = np.full(100, 3.)
    cls = generate_random_cluster(distances)
    dist_mat = get_distances(cls)
    min_dist = dist_mat.min(axis=1)
    return np.allclose(min_dist[1:], distances)


# ------------------------ the following is deprecated!
class Flake:
    def __init__(self):
        self.o = [[0, 0, 0]]
        pass

    @property
    def coordination(self):
        return len(self.coo)

    def _coo(self, o=False):
        t = type(self.coo)
        if t == list:
            coo = self.coo
        elif t == np.ndarray or t == torch.Tensor:
            coo = self.coo.tolist()
        else:
            raise RuntimeError('unknown type in Flake')
        if o:
            return self.o + coo
        else:
            return coo

    def tensor(self, a=1., o=False):
        return torch.as_tensor(self._coo(o=o)).type(torch.zeros(1).type()) * a

    def array(self, a=1., o=False):
        return np.asarray(self._coo(o=o), dtype=np.zeros((1,)).dtype) * a

    def show(self):
        from theforce.util.flake import show_flake
        show_flake(self.array(), batch=False)


class SC(Flake):
    def __init__(self):
        super().__init__()
        self.coo = [[1, 0, 0], [-1, 0, 0],
                    [0, 1, 0], [0, -1, 0],
                    [0, 0, 1], [0, 0, -1]]


class BCC(Flake):
    def __init__(self):
        super().__init__()
        self.coo = [[1, 1, 1], [-1, -1, -1],
                    [1, 1, -1], [-1, -1, 1],
                    [1, -1, 1], [-1, 1, -1],
                    [-1, 1, 1], [1, -1, -1]]


class FCC(Flake):
    def __init__(self):
        super().__init__()
        self.coo = [[1, 0, 0], [-1, 0, 0],
                    [0, 1, 0], [0, -1, 0],
                    [0, 0, 1], [0, 0, -1],
                    [1, 1, 1], [-1, -1, -1],
                    [1, 1, -1], [-1, -1, 1],
                    [1, -1, 1], [-1, 1, -1],
                    [-1, 1, 1], [1, -1, -1]]


class Hex(Flake):
    def __init__(self):
        super().__init__()
        a = 1.0
        c = a * np.sqrt(2/3)
        cell = np.array([[a, 0, 0],
                         [a/2, a*np.sqrt(3)/2, 0],
                         [0, 0, c]])
        coo = [[1, 0, 0], [0, 1, 0], [-1, 1, 0], [-1, 0, 0], [0, -1, 0], [1, -1, 0],
               [1./3, 1./3, 1], [-2./3, 1./3, 1], [1./3, -2./3, 1],
               [1./3, 1./3, -1], [-2./3, 1./3, -1], [1./3, -2./3, -1]]
        self.coo = np.einsum('ij,jk->ik', np.array(coo), cell)


class ZB(Flake):
    def __init__(self):
        super().__init__()
        self.coo = [[-1, -1, 1], [1, 1, 1],
                    [1, -1, -1], [-1, 1, -1]]


AllMyFlakes = [SC, BCC, FCC, Hex, ZB]


def lennard_jones(xyz, sigma=1.0):
    r = np.sqrt((xyz**2).sum(axis=1)) / sigma
    pot = 4*(1/r**12 - 1/r**6).sum()
    forces_on_environ = 4*(12/r**14-6/r**8)[:, None] * xyz/sigma
    force_on_center = -forces_on_environ.sum(axis=0)
    return pot/2, force_on_center


# ------------------------------------------ flakes
def cubic_flake(a=1.0, centre=True):
    if centre:
        trans = [[0, 0, 0]]
    else:
        trans = []
    trans += [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [-1, 0, 0],
              [0, -1, 0],
              [0, 0, -1]]
    atoms = np.array(trans) * a
    return atoms


def hexagonal_flake(a=1.0, centre=False):
    c = a * np.sqrt(2/3)
    cell = np.array([[a, 0, 0],
                     [a/2, a*np.sqrt(3)/2, 0],
                     [0, 0, c]])
    if centre:
        trans = [[0, 0, 0]]
    else:
        trans = []
    trans += [[1, 0, 0], [0, 1, 0], [-1, 1, 0], [-1, 0, 0], [0, -1, 0], [1, -1, 0],
              [1./3, 1./3, 1], [-2./3, 1./3, 1], [1./3, -2./3, 1],
              [1./3, 1./3, -1], [-2./3, 1./3, -1], [1./3, -2./3, -1]]
    t = np.array(trans)
    atoms = np.einsum('ij,jk->ik', t, cell)
    return atoms


# ------------------------------------------ data generators
def from_flake(flake, num, eps, potential=lennard_jones):
    """ eps is a random distortion """
    xyz = np.stack([flake + np.random.uniform(-eps, eps, size=flake.shape)
                    for _ in range(num)])
    energies, forces = (np.stack(a) for a in
                        zip(*[potential(c) for c in xyz]))
    return xyz, energies, forces


# ------------------------------------------ visualization (mayavi)
def show_flake(xyz, rc=3, d=0.2,  batch=True):
    if batch:
        _xyz = xyz
    else:
        _xyz = [xyz]
    D = rc * d
    # %gui qt
    from mayavi import mlab
    for a in _xyz:
        r = np.sqrt((a**2).sum(axis=-1))
        s = np.ones_like(r) * D
        mlab.points3d(*a.T, s, scale_factor=1)
        #  center and cutoff
        #mlab.points3d(0, 0, 0, D, color=(0, 0, 0), scale_factor=1)
        mlab.points3d(0, 0, 0, extent=[-rc, rc, -rc, rc, -rc, rc],
                      opacity=0.1, resolution=80)
        mlab.show()


if __name__ == '__main__':

    assert test_generate_random_cluster()

    xyz = hexagonal_flake()
    xyz, energies, forces = from_flake(xyz, 3, eps=0.1)
    #show_flake(xyz, rc=1.5, d=0.3)

    xyz = cubic_flake()
    xyz, energies, forces = from_flake(xyz, 3, eps=0.1)
    #show_flake(xyz, rc=1.5, d=0.3)
