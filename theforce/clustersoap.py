
# coding: utf-8

# In[ ]:


import numpy as np
from ase.neighborlist import NewPrimitiveNeighborList


class ClusterSoap:
    """ 
    A cluster is defined by its box=(pbc, cell)
    as well as positions of its atoms.
    It uses ase to create neighbors list and then
    constructs the descriptor vectors for each atom.
    It also calculates derivatives of descriptors wrt
    positions of atoms.
    """

    def __init__(self, soap):
        """ 
        Needs an instance of sesoap class.
        """
        self.soap = soap
        self.neighbors = NewPrimitiveNeighborList(soap.rc, skin=0.0, self_interaction=False,
                                                  bothways=True)

    def descriptors(self, pbc, cell, positions, sumj=True):
        """ 
        Inputs: pbc, cell, positions 
        Returns: p, q, pairs
        p: descripter vector which is per-atom.
        q: derivatives of p wrt coordinates of atoms, which is per-atom if 
        sumj is True, per-pair if sumj is False.
        --------------------------------------------------------------------
        If sumj is False, q[k,l,m] is a sparse matrix where k refers to k'th 
        pair (in pairs), l counts the dimension of the descriptor, and m counts 
        the 3 Cartesian axes.
        If sumj is True, the only difference is that k counts the atoms.
        ------------------------------------------------------------------
        Notice that when sumj=False, q is a sparse matrix.
        Otherwise all arrays are full (zeros will be passed if no atoms are
        present in the environment).
        Thus p, pairs will be the same either way.
        """
        self.neighbors.build(pbc, cell, positions)
        n = positions.shape[0]
        _p = []
        _q = []
        pairs = []
        for k in range(n):
            indices, offsets = self.neighbors.get_neighbors(k)
            if indices.shape[0] > 0:
                env = positions[indices] + np.einsum('ik,kj->ij', offsets, cell)                     - positions[k]
                a, b = self.soap.derivatives(env, sumj=sumj)
                _p += [a]
                _q += [b]
                pairs += [(k, j) for j in indices]
            else:
                _p += [np.zeros(shape=self.soap.dim)]
                if sumj:
                    _q += [np.zeros(shape=(self.soap.dim, 3))]
        p = np.asarray(_p)
        if sumj:
            q = np.asarray(_q)
        elif len(_q) > 0:
            q = np.transpose(np.concatenate(_q, axis=1), axes=[1, 0, 2])
        else:
            q = np.array([])
        return p, q, pairs


def test_if_works():
    import numpy as np
    from ase import Atoms
    from theforce.util.flake import hexagonal_flake
    from theforce.sesoap import sesoap
    from theforce.radial_funcs import quadratic_cutoff

    flake = hexagonal_flake(a=1.1)
    positions = np.concatenate(([[0, 0, 0]], flake)) + np.array([3., 3., 3.])
    atoms = Atoms(positions=positions, cell=[6., 6., 6.], pbc=True)

    lmax, nmax, cutoff = 4, 4, 1.1000000001
    soap = sesoap(lmax, nmax, quadratic_cutoff(cutoff))
    csoap = ClusterSoap(soap)
    p, q, pairs = csoap.descriptors(
        atoms.pbc, atoms.cell, atoms.positions, sumj=False)
    print('it works!', p.shape, q.shape, len(pairs))
    
    import nglview
    return nglview.show_ase(atoms)


if __name__ == '__main__':

    view = test_if_works()

