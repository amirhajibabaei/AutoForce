
# coding: utf-8

# In[ ]:


import numpy as np
from ase.neighborlist import NewPrimitiveNeighborList
from torch.autograd import Function


class ClusterSoap:
    """ 
    A cluster is defined by its box=(pbc, cell)
    as well as positions of its atoms.
    It uses ase to create neighbors list and then
    constructs the descriptor vectors for each atom.
    It also calculates derivatives of descriptors wrt
    positions of atoms.
    """

    def __init__(self, soap, sorted=False):
        """ 
        Needs an instance of sesoap class.
        """
        self.soap = soap
        self.neighbors = NewPrimitiveNeighborList(soap.rc, skin=0.0, self_interaction=False,
                                                  bothways=True, sorted=sorted)

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
        self.neighbors.build(pbc, cell, positions)  # TODO: update maybe faster
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


class TorchSoap(Function):
    """
    A wrapper around the ClusterSoap.
    The key variable is xyz (in the forward method) which should be
    a torch.tensor object and it may have xyz.requires_grad=True.
    """

    @staticmethod
    def forward(ctx, pbc, cell, xyz, csoap):
        """ csoap is an instance of ClusterSoap """
        _xyz = xyz.detach().numpy()
        _p, _q, _ = csoap.descriptors(pbc, cell, _xyz)
        p = torch.as_tensor(_p, dtype=xyz.dtype)
        q = torch.as_tensor(_q, dtype=xyz.dtype)
        ctx.save_for_backward(q)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        q, = ctx.saved_tensors
        grad = torch.einsum('ij,ijk->ik', grad_output, q)
        return None, None, grad, None


torchsoap = TorchSoap.apply


def test_if_works():
    import numpy as np
    from ase import Atoms
    from theforce.util.flake import hexagonal_flake
    from theforce.descriptor.sesoap import SeSoap
    from theforce.descriptor.radial_funcs import quadratic_cutoff

    a = 1.0
    lmax, nmax, cutoff = 6, 6, a+1e-3
    soap = SeSoap(lmax, nmax, quadratic_cutoff(cutoff))
    csoap = ClusterSoap(soap)

    pbc = True
    cell = np.array([10, 10, 10])*a
    center = cell/2
    flake = hexagonal_flake(a=a, centre=True)

    positions_a = flake + center
    atoms_a = Atoms(positions=positions_a, cell=cell, pbc=pbc)
    p_a, q_a, pairs_a = csoap.descriptors(atoms_a.pbc, atoms_a.cell,
                                          atoms_a.positions, sumj=True)

    positions_b = flake - center*0.33
    atoms_b = Atoms(positions=positions_b, cell=cell, pbc=pbc)
    p_b, q_b, pairs_b = csoap.descriptors(atoms_b.pbc, atoms_b.cell,
                                          atoms_b.positions, sumj=True)

    print(np.allclose(p_a-p_b, 0.0),
          np.allclose(q_a-q_b, 0.0), (q_a-q_b).max())


if __name__ == '__main__':

    view = test_if_works()

