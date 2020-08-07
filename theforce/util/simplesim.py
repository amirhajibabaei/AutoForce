from ase.neighborlist import NeighborList
import numpy as np


class SimpleSim:

    def __init__(self, atoms, cutoff=5., alpha=0.2):
        nl = NeighborList(len(atoms)*[cutoff/2], skin=0.0,
                          self_interaction=False, bothways=True)
        nl.update(atoms)
        data = []
        for a in range(len(atoms)):
            n, off = nl.get_neighbors(a)
            cells = (off[..., None]*atoms.cell).sum(axis=1)
            r = atoms.positions[n] - atoms.positions[a] + cells
            d = np.linalg.norm(r, axis=1)
            data += [(n, d)]

        self.data = data
        self.atoms = atoms
        self.rc = cutoff
        self.alpha = alpha

    def _kern(self, i, j):
        n, d = self.data[i]
        nn, dd = self.data[j]
        z = self.atoms.numbers[n]
        zz = self.atoms.numbers[nn]
        value = 0.
        for s in set(z).union(set(zz)):
            i = z == s
            ii = zz == s
            f = np.exp(-((d[i][:, None] - dd[ii][None])/self.alpha)**2)
            c = ((1.-d[i]/self.rc)**2)[:, None]*((1.-dd[ii]/self.rc)**2)[None]
            k = (f*c).sum()
            value += k
        return value

    def kern(self, i, j):
        a = self._kern(i, j)
        b = self._kern(i, i)
        c = self._kern(j, j)
        return a/np.sqrt(b*c)

    def __call__(self, i, j, threshold=0.95):
        return self.kern(i, j) > threshold
