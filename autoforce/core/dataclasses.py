# +
from torch import Tensor
from typing import List, Any, Optional


class PotData:

    __slots__ = ('energy', 'forces')

    def __init__(self,
                 *,
                 energy: Optional[Tensor] = None,
                 forces: Optional[Tensor] = None
                 ) -> None:
        """
        Self explanatory.

        """
        self.energy = energy
        self.forces = forces


class Conf:
    """
    An atomic configuration.

    Similar to ase.Atoms, except the data
    are stored as torch.Tensor instead of
    np.ndarray.


    Keywords:
    numbers      atomic numbers
    positions    self explanatory
    cell         self explanatory
    pbc          periodic boundary conditions
    potential    energy & forces

    """

    __slots__ = ('numbers',
                 'positions',
                 'cell',
                 'pbc',
                 'potential',
                 '_isolated_atoms',
                 '_cached_LocalEnv',
                 '_cached_LocalDes')

    def __init__(self,
                 numbers: Tensor,
                 positions: Tensor,
                 cell: Tensor,
                 pbc: List[bool],
                 potential: Optional[PotData] = None
                 ) -> None:
        """
        Self explanatory.

        """

        if positions.requires_grad != cell.requires_grad:
            raise RuntimeError('requires_grad should be the '
                               'same for positions and cell!'
                               )

        self.numbers = numbers
        self.positions = positions
        self.cell = cell
        self.pbc = pbc

        if potential is None:
            potential = PotData()
        self.potential = potential

        self._isolated_atoms = None
        self._cached_LocalEnv = None
        self._cached_LocalDes = None

    @property
    def requires_grad(self) -> bool:
        return self.positions.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self.positions.requires_grad = value
        self.cell.requires_grad = value


class LocalEnv:
    """
    Local chemical environment (LCE) of an atom.

    Keywords:
    central atom    the atom for which the LocalEnv
                    is created
    neighborhood    atoms in the neighborhood of
                    the central atom (|rij| < cutoff)

    On wij:
    wij are weights of the neighborhood atoms which
    are typically set by a Cutoff_fn e.g.:
        wij = (1-rij/cutoff)**2
    but, they can also be utilized for generalized
    weighing mechanisms. Descriptor functions should
    behave such that if for a neighbor wij=0, the
    result should be the same as eliminating that
    neighbor from the LocalEnv.

    """

    __slots__ = ('index',
                 'number',
                 'numbers',
                 'rij',
                 'wij')

    def __init__(self,
                 index: Tensor,
                 number: Tensor,
                 numbers: Tensor,
                 rij: Tensor,
                 wij: Optional[Tensor] = None
                 ) -> None:
        """
        index    index of the central atom in the
                 Conf it belongs to.
        number   atomic number of the central atom
        numbers  atomic numbers of the neighborhood atoms
        rij      coordinates of the neighborhood atoms
                 relative to the central atom
        wij      weights of the neighborhood atoms

        """

        self.index = index
        self.number = number
        self.numbers = numbers
        self.rij = rij
        self.wij = wij


class LocalDes:
    """
    Local descriptor vector, feature vector, or
    fingerprints for a LocalEnv.

    It is often the output of a "descriptor
    function" which can opaquely handle its
    own data structures via "tensors" and "meta".

    "tensors" are reserved for the data which
    are derived from atomic positions and their
    gradients maybe tracked.

    "meta" are arbitrary auxiliary data.

    "index" is the index of the LocalEnv from which
    the descriptor is derived from. If None,
    it is automatically handled.

    "species" is the species of the model which
    will handle the descriptor. If None, it is
    automatically handled. In most cases species
    are automatically set to the atomic number
    of the central atom (i.e. a model per atom
    type). In rare cases one might be interested
    in mixing several atom types in a single model
    which can be achieved by considering them
    as the same species.

    Thus, in the contex of this package,
    "species" and "atomic numbers" refer to
    two different concepts.


    TODO: explain orientation attribute.

    """

    __slots__ = ('tensors',
                 'meta',
                 'index',
                 'species',
                 'norm',
                 '_cached_orientation')

    def __init__(self,
                 *tensors: Any,  # TODO: Any <- Tuple[Tensor, ...]
                 meta: Any = None,
                 index: Optional[int] = None,
                 species: Optional[int] = None,
                 norm: Optional[Tensor] = None
                 ) -> None:
        """
        tensors    a tuple of tensors (main data)
        meta       arbitrary auxiliary data
        index      index of the central atom
        species    species of the descriptor
        norm       norm of the descriptor

        """

        self.tensors = tensors
        self.meta = meta
        self.index = index
        self.species = species
        self.norm = norm

        self._cached_orientation = []

    def detach(self):  # TODO: -> LocalDes
        tensors = (t.detach() for t in self.tensors)
        detached = LocalDes(*tensors,
                            meta=self.meta,
                            index=self.index,
                            species=self.species,
                            norm=self.norm.detach())
        detached._cached_orientation = [t.detach() for t in
                                        self._cached_orientation]
        return detached
