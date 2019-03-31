
# coding: utf-8

# In[ ]:


import numpy as np
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList


class AtomsSoap:
    """ connects the sesoap descriptor and the popular ase.Atoms object """

    def __init__(self, soap):
        """ 
        Cutoff is the only parameter this needs.
        It will use soap.rc as cutoff.
        """
        self.soap = soap
        self.neighbors = NeighborList(soap.rc, skin=0.0, self_interaction=False,
                                      bothways=True, primitive=NewPrimitiveNeighborList)

    def descriptors(self, atoms, sumj=True):
        """ 
        Inputs: ase Atoms object
        Returns: p, q, pairs
        p: descripter vector which is per-atom.
        q: derivatives of p wrt coordinates of atoms, which is per-atom if 
        sumj is True, per-pair if sumj is False.
        --------------------------------------------------------------------
        If sumj is False, q[k,l,m] is a sparse matrix where k refers to k'th 
        pair (in pairs), l counts the dimension of the descriptor, and m counts 
        3D Cartesian axes.
        If sumj is True, the only difference is that k counts the atoms.
        ------------------------------------------------------------------
        Notice that when sumj=False, q is a sparse matrix.
        Otherwise all arrays are full (zeros will be passed if no atoms are
        present in env).
        Thus p, pairs will be the same either way.
        """
        self.neighbors.update(atoms)
        n = atoms.get_number_of_atoms()
        _p = []
        _q = []
        pairs = []
        for k in range(n):
            indices, offsets = self.neighbors.get_neighbors(k)
            if indices.shape[0] > 0:
                env = atoms.positions[indices] +                     np.einsum('ik,kj->ij', offsets, atoms.cell) -                     atoms.positions[k]
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

