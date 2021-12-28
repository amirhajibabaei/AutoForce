# +
from torch import Tensor
from typing import List, Any, Optional


class Data:

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
    data         energy & forces

    """

    __slots__ = ('numbers', 'positions', 'cell', 'pbc', 'data')

    def __init__(self,
                 numbers: Tensor,
                 positions: Tensor,
                 cell: Tensor,
                 pbc: List[bool],
                 data: Optional[Data] = None
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

        if data is None:
            data = Data()
        self.data = data

    @property
    def requires_grad(self) -> bool:
        return self.positions.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self.positions.requires_grad = value
        self.cell.requires_grad = value


class Environ:
    """
    Local chemical environment (LCE) of an atom.

    Keywords:
    central atom    the atom for which the Environ
                    is created
    neighborhood    atoms in the neighborhood of
                    the central atom (|rij| < cutoff)

    """

    __slots__ = ('index', 'number', 'numbers', 'rij', 'wij')

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
                 (obtained by a smooth cutoff function)

        """

        self.index = index
        self.number = number
        self.numbers = numbers
        self.rij = rij
        self.wij = wij


class Descriptor:
    """
    Descriptor vector, feature vector, or
    fingerprints for a LCE.

    It is often the output of a "descriptor
    function" which can opaquely handle its
    own data structures via "tensors" and "meta".

    "tensors" are reserved for the data which
    are derived from atomic positions and their
    gradients are tracked.

    "meta" are arbitrary auxiliary data.

    "index" is the index of the LCE from which
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

    """

    __slots__ = ('tensors', 'meta', 'index', 'species', 'norm')

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

    def detach(self):  # TODO: -> Descriptor
        tensors = (t.detach() for t in self.tensors)
        detached = Descriptor(*tensors,
                              meta=self.meta,
                              index=self.index,
                              species=self.species,
                              norm=self.norm.detach())
        return detached
