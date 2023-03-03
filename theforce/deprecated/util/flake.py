# +
import numpy as np
import torch


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
            raise RuntimeError("unknown type in Flake")
        if o:
            return self.o + coo
        else:
            return coo

    def tensor(self, a=1.0, o=False):
        return torch.as_tensor(self._coo(o=o)).type(torch.zeros(1).type()) * a

    def array(self, a=1.0, o=False):
        return np.asarray(self._coo(o=o), dtype=np.zeros((1,)).dtype) * a

    def show(self):
        from theforce.util.flake import show_flake

        show_flake(self.array(), batch=False)


class SC(Flake):
    def __init__(self):
        super().__init__()
        self.coo = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]


class BCC(Flake):
    def __init__(self):
        super().__init__()
        self.coo = [
            [1, 1, 1],
            [-1, -1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
        ]


class FCC(Flake):
    def __init__(self):
        super().__init__()
        self.coo = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 1],
            [-1, -1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
        ]


class Hex(Flake):
    def __init__(self):
        super().__init__()
        a = 1.0
        c = a * np.sqrt(2 / 3)
        cell = np.array([[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
        coo = [
            [1, 0, 0],
            [0, 1, 0],
            [-1, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [1, -1, 0],
            [1.0 / 3, 1.0 / 3, 1],
            [-2.0 / 3, 1.0 / 3, 1],
            [1.0 / 3, -2.0 / 3, 1],
            [1.0 / 3, 1.0 / 3, -1],
            [-2.0 / 3, 1.0 / 3, -1],
            [1.0 / 3, -2.0 / 3, -1],
        ]
        self.coo = np.einsum("ij,jk->ik", np.array(coo), cell)


class ZB(Flake):
    def __init__(self):
        super().__init__()
        self.coo = [[-1, -1, 1], [1, 1, 1], [1, -1, -1], [-1, 1, -1]]


AllMyFlakes = [SC, BCC, FCC, Hex, ZB]


def lennard_jones(xyz, sigma=1.0):
    r = np.sqrt((xyz**2).sum(axis=1)) / sigma
    pot = 4 * (1 / r**12 - 1 / r**6).sum()
    forces_on_environ = 4 * (12 / r**14 - 6 / r**8)[:, None] * xyz / sigma
    force_on_center = -forces_on_environ.sum(axis=0)
    return pot / 2, force_on_center


# ------------------------------------------ flakes
def cubic_flake(a=1.0, centre=True):
    if centre:
        trans = [[0, 0, 0]]
    else:
        trans = []
    trans += [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    atoms = np.array(trans) * a
    return atoms


def hexagonal_flake(a=1.0, centre=False):
    c = a * np.sqrt(2 / 3)
    cell = np.array([[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    if centre:
        trans = [[0, 0, 0]]
    else:
        trans = []
    trans += [
        [1, 0, 0],
        [0, 1, 0],
        [-1, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [1, -1, 0],
        [1.0 / 3, 1.0 / 3, 1],
        [-2.0 / 3, 1.0 / 3, 1],
        [1.0 / 3, -2.0 / 3, 1],
        [1.0 / 3, 1.0 / 3, -1],
        [-2.0 / 3, 1.0 / 3, -1],
        [1.0 / 3, -2.0 / 3, -1],
    ]
    t = np.array(trans)
    atoms = np.einsum("ij,jk->ik", t, cell)
    return atoms


# ------------------------------------------ data generators
def from_flake(flake, num, eps, potential=lennard_jones):
    """eps is a random distortion"""
    xyz = np.stack(
        [flake + np.random.uniform(-eps, eps, size=flake.shape) for _ in range(num)]
    )
    energies, forces = (np.stack(a) for a in zip(*[potential(c) for c in xyz]))
    return xyz, energies, forces


# ------------------------------------------ visualization (mayavi)
def show_flake(xyz, rc=3, d=0.2, batch=True):
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
        # mlab.points3d(0, 0, 0, D, color=(0, 0, 0), scale_factor=1)
        mlab.points3d(
            0, 0, 0, extent=[-rc, rc, -rc, rc, -rc, rc], opacity=0.1, resolution=80
        )
        mlab.show()


if __name__ == "__main__":

    xyz = hexagonal_flake()
    xyz, energies, forces = from_flake(xyz, 3, eps=0.1)
    # show_flake(xyz, rc=1.5, d=0.3)

    xyz = cubic_flake()
    xyz, energies, forces = from_flake(xyz, 3, eps=0.1)
    # show_flake(xyz, rc=1.5, d=0.3)
