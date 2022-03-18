# +
from autoforce.core import NeighborList
from ase.neighborlist import primitive_neighbor_list
from numpy import ndarray
from typing import List, Dict


class ASE_NeighborList(NeighborList):

    def get_neighborlist(self,
                         cutoff: Dict,
                         pbc: List[bool],
                         cell: ndarray,
                         positions: ndarray,
                         atomic_numbers: ndarray
                         ) -> (ndarray, ndarray, ndarray):

        i, j, sij = primitive_neighbor_list('ijS',
                                            pbc,
                                            cell,
                                            positions,
                                            cutoff,
                                            numbers=atomic_numbers)

        return i, j, sij
