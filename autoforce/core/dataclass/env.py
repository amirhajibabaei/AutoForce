# +
from torch import Tensor


class LocalEnv:
    """
    Local chemical environment (LCE) of an atom.

    Keywords:
    central atom    the atom for which the LocalEnv
                    is created
    neighborhood    atoms in the neighborhood of
                    the central atom (|rij| < cutoff)

    """

    __slots__ = ('index',
                 'number',
                 'neighbors',
                 'numbers',
                 'rij',
                 'dij',
                 '_cached_descriptors')

    def __init__(self,
                 index: Tensor,
                 number: Tensor,
                 neighbors: Tensor,
                 numbers: Tensor,
                 rij: Tensor,
                 ) -> None:
        """
        index    index of the central atom in the
                 Conf it belongs to.
        number   atomic number of the central atom
        numbers  atomic numbers of the neighborhood atoms
        rij      coordinates of the neighborhood atoms
                 relative to the central atom

        """

        self.index = index
        self.number = number
        self.neighbors = neighbors
        self.numbers = numbers
        self.rij = rij
        self.dij = rij.norm(dim=1)

        # cache
        self._cached_descriptors = []
