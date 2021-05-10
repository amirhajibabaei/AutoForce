import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from ase.neighborlist import NeighborList
import warnings


def check_total_area_volume(vor, data):
    """
    Tests if the split data of the Voronoi cell adds 
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
    rij: Relative coordinates of an atom's neighbors (shape=[:, 3]).
         It is assumed that the coordinates of the central 
         atom itself (=[0, 0, 0]) is not included in rij.

    The Voronoi cell (around origin), is constructed
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


def get_vor_neighborhood(atoms, cutoff):
    """
    atoms: ase.Atoms object
    cutoff: Consider i, j as neighbors if |rj-ri| < cutoff

    Returns: I, J, A, V
    I, J: indices of atoms who share a Voronoi facet (neighbors)
    A: The surface area of facet
    V: The volume around I spanned by J

    Note:
    1) i, j and j, i repeated.
    2) cutoff should be large enough to enclose all Voronoi neighbors.
    3) sum(V) == total volume. If not, a warning is raised.
       In that case, increase the cutoff.
    """
    N = atoms.get_global_number_of_atoms()
    nl = NeighborList(N*[cutoff/2], skin=0., bothways=True,
                      self_interaction=False)
    nl.update(atoms)
    I = []
    J = []
    A = []
    V = []
    for i in range(N):
        j, off = nl.get_neighbors(i)
        off = (off[..., None]*atoms.cell).sum(axis=1)
        # off is the offsets due to periodic boundary condition
        rij = atoms.positions[j] - atoms.positions[i] + off
        data = vor_local(rij)
        _j, areas, vols = zip(*data)
        I.extend(len(_j)*[i])
        J.extend(j[list(_j)].tolist())
        A.extend(areas)
        V.extend(vols)
    I, J, A, V = (np.array(buff) for buff in (I, J, A, V))
    if not np.isclose(V.sum(), atoms.get_volume()):
        warnings.warn(
            'Voronoi volumes did not add up to total volume: try larger cutoff!')
    return I, J, A, V
