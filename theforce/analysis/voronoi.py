# +
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from ase.neighborlist import NeighborList
import warnings


class VorNei:

    def __init__(self, i, j, zi, zj, rij, aij, vij):
        """
        i:   index of the central atom
        j:   indices for Voronoi neighbors of i
        zi:  atomic number for i
        zj:  atomic numbers for j
        rij: rj-ri
        aij: area of ij facet
        vij: volume around i spanned by ij facet,

        * in periodic cells sum vij = total volume

                      j
                      |
                      |
                      |
                xxxx-aij-xxxx
                x .   |   . x
                x  . vij .  x
                x   . | .   x
          m ----x---- i ----x---- k
                x     |     x
                x     |     x
                x     |     x
                xxxxxxxxxxxxx
                      |
                      |
                      |
                      l
        """
        self.i = i
        self.j = j
        self.zi = zi
        self.zj = zj
        self.rij = rij
        self.aij = aij
        self.vij = vij


def check_total_area_volume(vor, data):
    """
    tests if the split data of the Voronoi cell adds
    up to total area/volume.
    """
    reg = vor.point_region[0]
    indices = vor.regions[reg]
    test = ConvexHull(vor.vertices[indices])
    i, A, V = zip(*data)
    testA = np.isclose(test.volume, sum(V))
    testV = np.isclose(test.area, sum(A))
    return testA, testV


def vor_local(rij, test=False):
    """
    rij: relative coordinates of an atom's neighbors (shape=[:, 3]).
         it is assumed that the coordinates of the central
         atom itself (=[0, 0, 0]) is not included in rij.

    the Voronoi cell (around origin), is constructed
    and the indices, areas, and volumes corresponding to
    each facet/neighbor are returned:

        data = [(i1, area1, vol1), ...]
        indices, areas, vols = zip(*data)
    """
    dim = rij.shape[1]
    orig = np.zeros((1, dim))
    # add origin to the beginning of rij -> c_rij
    c_rij = np.r_[orig, rij]
    vor = Voronoi(c_rij)
    data = []
    # loop over nn pairs
    for i, (_a, _b) in enumerate(vor.ridge_points):
        a, b = sorted((_a, _b))
        # only interested in neighbors of orig
        if a == 0:
            # vert: vertices for the face of vor cell
            # that devides the space between a, b
            vert = vor.ridge_vertices[i]
            # face: coordinates of the face
            face = vor.vertices[vert]
            # ph: orig+face polyhedron
            ph = np.r_[orig, face]
            vol = ConvexHull(ph).volume
            # h: length of the line segment from origin to the face
            h = np.linalg.norm(vor.points[b])/2
            # area of the face is obtained from vol and h
            area = dim*vol/h
            # note below: b-1, because c_rij is used instead of rij
            data.append((b-1, area, vol))
    if test:
        assert all(check_total_area_volume(vor, data))
    return data


def get_voronoi_neighbors(atoms, cutoff):
    """
    atoms:  ase.Atoms object
    cutoff: limits the neighbor search to |rj-ri| < cutoff,
            used only for faster search, it should be large
            enough so that the output doesn't depend on cutoff.

    returns: a list of VorNei objects

    * notes:
    1) cutoff should be large enough to enclose all Voronoi neighbors.
    2) sum(vij) == total volume. If not, a warning is raised,
       in that case, increase the cutoff.
    """
    N = atoms.get_global_number_of_atoms()
    nl = NeighborList(N*[cutoff/2], skin=0., bothways=True,
                      self_interaction=False)
    nl.update(atoms)
    vornei = []
    vol = 0
    for i in range(N):
        j, off = nl.get_neighbors(i)
        off = (off[..., None]*atoms.cell).sum(axis=1)
        # off is the offsets due to periodic boundary condition
        rij = atoms.positions[j] - atoms.positions[i] + off
        data = vor_local(rij)
        _j, areas, vols = zip(*data)
        _j = list(_j)
        j_vor = j[_j]
        zi = atoms.numbers[i]
        zj = atoms.numbers[j_vor]
        vor_i = VorNei(i, j_vor, zi, zj, rij[_j], areas, vols)
        vornei.append(vor_i)
        vol += sum(vols)
    if not np.isclose(vol, atoms.get_volume()):
        warnings.warn(
            'Voronoi volumes did not add up to total volume: try larger cutoff!')
    return vornei
