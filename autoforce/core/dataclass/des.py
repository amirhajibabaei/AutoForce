# +
from torch import Tensor
from typing import Any, Optional


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
