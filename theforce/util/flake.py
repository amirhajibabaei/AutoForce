# +
import numpy as np
from numpy.linalg import norm


def generate_random_cluster(n, d, dim=3):
    """
    Generates a random cluster of n points such that
    all nearest neighbor distances are equal to d.

    n: number of points
    d: nearest neighbor distances
    dim: Cartesian dimensions
    """

    def random_unit_vector(dim=3):
        u = np.random.uniform(-1., 1., size=dim)
        u /= norm(u)
        return u

    c = np.zeros((1, dim))
    for _ in range(1, n):
        # u: random direction
        u = random_unit_vector(dim)
        # p: projection of all points on u-axis
        p = (u*c).sum(axis=1)
        # first collision -> x*u
        s = np.argsort(p)[::-1]
        for j, k in enumerate(s):
            # y: min distance of point k from u-axis
            y = norm(c[k]-p[k]*u)
            if y <= d:
                x = p[k] + np.sqrt(d**2-y**2)
                break
        # check the distance with other nearby points
        for k in s[j:]:
            if x-p[k] > d:
                break
            # y: distance of point k with the new point (x*u)
            y = norm(c[k]-x*u)
            if y < d:
                z = norm(c[k]-p[k]*u)
                x = p[k] + np.sqrt(d**2-z**2)
        # append x*u to the cluster
        c = np.r_[c, x*u.reshape(1, dim)]
    return c


def test_generate_random_cluster(n=1000, d=1., dim=3):
    cls = generate_random_cluster(n, d, dim=dim)
    dmat = np.linalg.norm(cls[None]-cls[:, None], axis=-1)
    # replace diganal zeros with a large number
    dmat += np.eye(n)*(dmat.max() + 1.)
    dmin = dmat.min(axis=1)
    return np.allclose(dmin, d)


if __name__ == '__main__':
    assert test_generate_random_cluster()
