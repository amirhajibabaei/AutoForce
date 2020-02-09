
import numpy as np
from ase.neighborlist import NewPrimitiveNeighborList
import torch
from torch.autograd import Function
from theforce.util.tensors import cat, split
from theforce.util.arrays import SparseArray


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

    def describe(self, pbc, cell, positions, iderive=True, jderive=True):
        """
        Inputs: pbc, cell, positions
        Returns: p, s
        p: descripter vector which is per-atom.
        s: derivatives of p wrt positions (a SparseArray is returned)
        """
        self.neighbors.build(pbc, cell, positions)  # TODO: update maybe faster
        n = positions.shape[0]
        p = []
        s = SparseArray(shape=(self.soap.dim, 0, 3))
        for k in range(n):
            indices, offsets = self.neighbors.get_neighbors(k)
            env = positions[indices] + np.einsum('ik,kj->ij', offsets, cell)                 - positions[k]
            a, b = self.soap.derivatives(env, sumj=False, normalize=False)
            p += [a]
            if iderive:
                s.add([k], [k], -b.sum(axis=1, keepdims=True))
            if jderive and indices.shape[0] > 0:
                s.add(k, indices, b)
        p = np.asarray(p)
        return p, s

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

    def descriptors_derivatives(self, pbc, cell, positions, sumj=False, jsorted=True, selfderive=True):
        """ 
        Inputs: pbc, cell, positions 
        Returns: p, q, i, j
        p: descripter vector which is per-atom.
        q: derivatives of p wrt coordinates of atoms, which is per-atom if 
        sumj is True, per-pair if sumj is False.
        --------------------------------------------------------------------
        * If sumj is True, jsorted will be assumed False.
        * If sumj is False q and indices are sequences with a length equal
          to the number of atoms is the cell.
        * If jsorted is True,
            q[j] indicates derivatives of p[i] wrt r_j for all i in
            the neighborhood of j.
        * If jsorted is False,
            q[i] indicates derivatives of p[i] wrt r_j for all j in
            the neighborhood of i.
        """
        self.neighbors.build(pbc, cell, positions)  # TODO: update maybe faster
        n = positions.shape[0]
        p, q, i, j, nj = [], [], [], [], []
        for k in range(n):
            indices, offsets = self.neighbors.get_neighbors(k)
            if indices.shape[0] > 0:
                env = positions[indices] + np.einsum('ik,kj->ij', offsets, cell)                     - positions[k]
                a, b = self.soap.derivatives(env, sumj=sumj)
                if not sumj and selfderive:
                    indices = np.concatenate([indices, [k]])
                    b = np.concatenate([b, -b.sum(axis=1, keepdims=True)],
                                       axis=1)
                p += [a]
                q += [b]
                i += [k]
                j += [indices]
                nj += [indices.shape[0]]
            else:
                p += [np.zeros(shape=self.soap.dim)]
                if sumj:
                    q += [np.zeros(shape=(self.soap.dim, 3))]
        p = np.asarray(p)
        if sumj:
            q = np.asarray(q)
        elif len(q) > 0:
            if jsorted:
                i = np.concatenate([a*[b] for a, b in zip(*[nj, i])])
                j = np.concatenate(j)
                k = np.argsort(j)
                sections = np.cumsum(nj)[:-1]
                i = np.split(i[k], indices_or_sections=sections)
                j = [np.unique(a) for a in np.split(
                    j[k], indices_or_sections=sections)]
                q = np.split(np.concatenate(q, axis=1)[:, k],
                             indices_or_sections=sections, axis=1)
                assert np.array([a.shape for a in j]).all()
                j = [a[0] for a in j]
        return p, q, i, j


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
        p, q, i, j = csoap.descriptors_derivatives(pbc, cell, _xyz, sumj=False,
                                                   jsorted=True)
        p = torch.as_tensor(p, dtype=xyz.dtype)
        # for backward
        n = torch.tensor(xyz.size(0))
        q, qspec = cat([torch.as_tensor(v, dtype=xyz.dtype) for v in q], 1)
        i, ispec = cat([torch.LongTensor(v) for v in i], 0)
        j = torch.tensor(j)
        ctx.save_for_backward(n, q, qspec, i, ispec, j)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        n, q, qspec, ii, ispec, jj = ctx.saved_tensors
        q = split(q, qspec)
        ii = split(ii, ispec)
        jj = jj.tolist()
        grads = torch.zeros(n, 3, dtype=q[0].dtype)
        for i, j,  dxi_drj in zip(*[ii, jj, q]):
            grads[j] = torch.einsum('ip,pim->m',
                                    grad_output[i], dxi_drj)
        return None, None, grads, None


torchsoap = TorchSoap.apply


def test_if_works():
    import numpy as np
    from ase import Atoms, Atom
    from theforce.util.flake import hexagonal_flake
    from theforce.descriptor.sesoap import SeSoap
    from theforce.descriptor.radial_funcs import quadratic_cutoff

    a = 1.0
    lmax, nmax, cutoff = 3, 3, a+1e-3
    soap = SeSoap(lmax, nmax, quadratic_cutoff(cutoff))
    csoap = ClusterSoap(soap)

    pbc = True
    cell = np.array([20, 10, 10])*a

    # test single atom
    positions = np.array([0, 0, 0]).reshape((1, 3))
    atoms = Atoms(positions=positions, cell=cell, pbc=pbc)
    p, q, i, j = csoap.descriptors_derivatives(atoms.pbc, atoms.cell, atoms.positions,
                                               sumj=False, jsorted=True)
    # test flake and its rigid translation
    center = cell/2
    flake = hexagonal_flake(a=a, centre=True)
    positions_a = flake + center
    atoms_a = Atoms(positions=positions_a, cell=cell, pbc=pbc)
    p_a, q_a, is_a, js_a = csoap.descriptors_derivatives(atoms_a.pbc, atoms_a.cell,
                                                         atoms_a.positions, sumj=True)
    positions_b = flake - center*0.33
    atoms_b = Atoms(positions=positions_b, cell=cell, pbc=pbc)
    p_b, q_b, is_b, js_b = csoap.descriptors_derivatives(atoms_b.pbc, atoms_b.cell,
                                                         atoms_b.positions, sumj=True)
    print(np.allclose(p_a-p_b, 0.0),
          np.allclose(q_a-q_b, 0.0), (q_a-q_b).max())

    p_ = csoap.descriptors(atoms_b.pbc, atoms_b.cell,
                           atoms_b.positions)
    print(np.allclose(p_a-p_, 0.0))

    # test system with one isolated atom
    atoms_b[-1].position = [0, 0, 0]
    p, q, i, j = csoap.descriptors_derivatives(atoms_b.pbc, atoms_b.cell, atoms_b.positions,
                                               sumj=False, jsorted=True)
    # for a, b in zip(q, i):
    #    print(a.shape, b.shape, b)

    xyz = torch.tensor(atoms_b.positions, requires_grad=True)
    p = torchsoap(atoms_b.pbc, atoms_b.cell, xyz, csoap)
    print(p.shape)
    a = (p**2).sum()
    a.backward()


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
    p1, q1, i1, j1 = csoap.descriptors_derivatives(atoms.pbc, atoms.cell, atoms.positions,
                                                   jsorted=False, sumj=False)
    dr = np.random.uniform(-delta, delta, size=atoms.positions.shape)
    atoms.positions += dr
    p2, q2, i2, r2 = csoap.descriptors_derivatives(atoms.pbc, atoms.cell, atoms.positions,
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

