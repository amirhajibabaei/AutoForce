# +
from torch import Tensor
from collections import defaultdict
import itertools
from typing import List, Dict, Any, Optional


class Target:

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

    Same role as ase.Atoms, except the data
    are stored as torch.Tensor instead of
    np.ndarray, for using the autograd.


    Keywords:
    numbers      atomic numbers
    positions    self explanatory
    cell         self explanatory
    pbc          periodic boundary conditions
    target       energy & forces

    """

    __slots__ = ('numbers',
                 'positions',
                 'cell',
                 'pbc',
                 'target',
                 'number_of_atoms',
                 'unique_counts',
                 '_cached_local_envs',
                 '_cached_isolated_atoms')

    def __init__(self,
                 numbers: Tensor,
                 positions: Tensor,
                 cell: Tensor,
                 pbc: List[bool],
                 target: Optional[Target] = None
                 ) -> None:
        """
        Self explanatory.

        """

        if positions.requires_grad != cell.requires_grad:
            raise RuntimeError('requires_grad should be the '
                               'same for positions and cell!'
                               )

        self.numbers = numbers
        self.number_of_atoms = int(numbers.numel())
        u, c = numbers.unique(return_counts=True)
        self.unique_counts = {int(a): int(b) for a, b in zip(u, c)}
        self.positions = positions
        self.cell = cell
        self.pbc = pbc

        if target is None:
            target = Target()
        self.target = target

        # cache
        self._cached_local_envs = None
        self._cached_isolated_atoms = None

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

    """

    __slots__ = ('index',
                 'number',
                 'numbers',
                 'rij',
                 'dij',
                 '_cached_descriptors')

    def __init__(self,
                 index: Tensor,
                 number: Tensor,
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
        self.numbers = numbers
        self.rij = rij
        self.dij = rij.norm(dim=1)

        # cache
        self._cached_descriptors = []


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

    """

    __slots__ = ('tensors',
                 'meta',
                 'index',
                 'species',
                 'norm',
                 '_cached_scalar_products')

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

        # cache
        self._cached_scalar_products = []

    def detach(self):  # TODO: -> LocalDes
        tensors = (t.detach() for t in self.tensors)
        detached = LocalDes(*tensors,
                            meta=self.meta,
                            index=self.index,
                            species=self.species,
                            norm=self.norm.detach())
        detached._cached_scalar_products = [[t.detach() for t in wrt]
                                            for wrt in self._cached_scalar_products]
        return detached


class Basis:

    __slots__ = ('index',
                 'descriptors',
                 'active')

    def __init__(self):
        self.index = None
        self.descriptors = defaultdict(list)
        self.active = defaultdict(list)

    def append(self, d: LocalDes) -> None:
        self.descriptors[d.species].append(d.detach())
        self.active[d.species].append(True)

    def count(self) -> Dict:
        return {s: a.count(True) for s, a in self.active.items()}

    def norms(self) -> Dict:
        return {s: [d.norm for d in itertools.compress(self.descriptors[s], a)]
                for s, a in self.active.items()}
