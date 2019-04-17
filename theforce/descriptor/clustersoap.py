#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from ase.neighborlist import NewPrimitiveNeighborList
from torch.autograd import Function


class ClusterSoap:
    """ 
    A cluster is defined by its box=(pbc, cell)
    as well as positions of its atoms.
    It uses ASE to create neighbors list and then
    constructs the descriptor vectors for each atom.
    It also calculates derivatives of descriptors wrt
    positions of atoms.
    """

    def __init__(self, soap):
        """ 
        Needs an instance of SeSoap class.
        """
        self.soap = soap
        self.neighbors = NewPrimitiveNeighborList(soap.rc, skin=0.0, self_interaction=False,
                                                  bothways=True, sorted=False)

    def descriptors(self, pbc, cell, positions):
        """ 
        Inputs: pbc, cell, positions 
        Returns: p 
        p is the descripter vector which is per-atom.
        """
        self.neighbors.build(pbc, cell, positions)  # TODO: update maybe faster
        n = positions.shape[0]
        _p = []
        for k in range(n):
            indices, offsets = self.neighbors.get_neighbors(k)
            if indices.shape[0] > 0:
                env = positions[indices] + np.einsum('ik,kj->ij', offsets, cell)                     - positions[k]
                _p += [self.soap.descriptor(env)]
            else:
                _p += [np.zeros(shape=self.soap.dim)]
        p = np.asarray(_p)
        return p

    def descriptors_derivatives(self, pbc, cell, positions, sumj=False, jsorted=True):
        """ 
        Inputs: pbc, cell, positions 
        Returns: p, q, indices
        p: descripter vector which is per-atom.
        q: derivatives of p wrt coordinates of atoms, which is per-atom if 
        sumj is True, per-pair if sumj is False.
        --------------------------------------------------------------------
        * If sumj is True, jsorted will be assumed False.
        * If sumj is False q and indices are sequences with a length equal
          to the number of atoms is the cell.
        * If jsorted is True,
            q[j] indicates derivatives of p[i] wrt r_j for all i in
            the neighborhood of j which are saved in indices[j].
        * If jsorted is False,
            q[i] indicates derivatives of p[i] wrt r_j for all j in
            the neighborhood of i which are saved in indices[i].
        """
        self.neighbors.build(pbc, cell, positions)  # TODO: update maybe faster
        n = positions.shape[0]
        _p = []
        _q = []
        js = []
        nj = []
        for k in range(n):
            indices, offsets = self.neighbors.get_neighbors(k)
            if indices.shape[0] > 0:
                env = positions[indices] + np.einsum('ik,kj->ij', offsets, cell)                     - positions[k]
                a, b = self.soap.derivatives(env, sumj=sumj)
                _p += [a]
                _q += [b]
                js += [indices]
                nj += [indices.shape[0]]
            else:
                _p += [np.zeros(shape=self.soap.dim)]
                if sumj:
                    _q += [np.zeros(shape=(self.soap.dim, 3))]
                js += [[]]
                nj += [0]
        p = np.asarray(_p)
        if sumj:
            q = np.asarray(_q)
        else:
            if jsorted:
                _is = np.concatenate([np.full_like(a, i)
                                      for i, a in enumerate(js)])
                _js = np.concatenate(js)
                k = np.argsort(_js)
                sections = np.cumsum(nj)[:-1]
                _is = np.split(_is[k], indices_or_sections=sections)
                q = np.split(np.concatenate(_q, axis=1)[:, k],
                             indices_or_sections=sections, axis=1)
            else:
                q = _q
        indices = (js if sumj or not jsorted else _is)
        return p, q, indices


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
        _p, _q, _ = csoap.descriptors_derivatives(pbc, cell, _xyz)
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
    lmax, nmax, cutoff = 3, 3, a+1e-3
    soap = SeSoap(lmax, nmax, quadratic_cutoff(cutoff))
    csoap = ClusterSoap(soap)

    pbc = True
    cell = np.array([10, 10, 10])*a
    center = cell/2
    flake = hexagonal_flake(a=a, centre=True)

    positions_a = flake + center
    atoms_a = Atoms(positions=positions_a, cell=cell, pbc=pbc)
    p_a, q_a, js_a = csoap.descriptors_derivatives(atoms_a.pbc, atoms_a.cell,
                                                   atoms_a.positions, sumj=True)

    positions_b = flake - center*0.33
    atoms_b = Atoms(positions=positions_b, cell=cell, pbc=pbc)
    p_b, q_b, js_b = csoap.descriptors_derivatives(atoms_b.pbc, atoms_b.cell,
                                                   atoms_b.positions, sumj=True)

    print(np.allclose(p_a-p_b, 0.0),
          np.allclose(q_a-q_b, 0.0), (q_a-q_b).max())

    p_ = csoap.descriptors(atoms_b.pbc, atoms_b.cell,
                           atoms_b.positions)
    print(np.allclose(p_a-p_, 0.0))

    #
    p, q, indices = csoap.descriptors_derivatives(atoms_b.pbc, atoms_b.cell, atoms_b.positions,
                                                  sumj=False, jsorted=True)
    # for a, b in zip(q, indices):
    #    print(a.shape, b.shape, b)


def test_grad():
    import numpy as np
    from ase import Atoms
    from theforce.util.flake import hexagonal_flake
    from theforce.descriptor.sesoap import SeSoap
    from theforce.descriptor.radial_funcs import quadratic_cutoff

    # csoap
    a = 1.0
    lmax, nmax, cutoff = 3, 3, a+1e-3
    soap = SeSoap(lmax, nmax, quadratic_cutoff(cutoff))
    csoap = ClusterSoap(soap)

    # atoms
    pbc = True
    cell = np.array([10, 10, 10])*a
    center = cell/2
    flake = hexagonal_flake(a=a, centre=True)
    positions = flake + center
    atoms = Atoms(positions=positions, cell=cell, pbc=pbc)

    # dp
    delta = 1e-5
    p1, q1, idx1 = csoap.descriptors_derivatives(atoms.pbc, atoms.cell, atoms.positions,
                                                 jsorted=False, sumj=False)
    dr = np.random.uniform(-delta, delta, size=atoms.positions.shape)
    atoms.positions += dr
    p2, q2, idx2 = csoap.descriptors_derivatives(atoms.pbc, atoms.cell, atoms.positions,
                                                 jsorted=False, sumj=False)

    dpr = np.concatenate([a.sum(axis=1, keepdims=True)
                          for a in q1], axis=1) * dr
    dp = dpr.sum(axis=(1, 2))
    # the point is that error scales with delta, to see this,
    # try dp with different orders of delta and calculate p2-p1
    test = np.allclose(p2-p1, dp, atol=10**3*delta)
    print(test)


if __name__ == '__main__':

    view = test_if_works()

    test_grad()

