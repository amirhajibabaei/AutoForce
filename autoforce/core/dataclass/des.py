# +
from torch import Tensor
from typing import Tuple, Dict, Optional
from autoforce.aliases import Descriptor_t


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

    __slots__ = ('descriptor',
                 'species',
                 'norm',
                 '_cached_scalar_products')

    def __init__(self,
                 descriptor: Descriptor_t,
                 index: Optional[int] = None,
                 species: Optional[int] = None,
                 norm: Optional[Tensor] = None
                 ) -> None:
        """
        descriptor   a dict of {key: tensor} (main data)
        species      species of the descriptor
        norm         norm of the descriptor

        """

        self.descriptor = descriptor
        self.species = species
        self.norm = norm

        # cache
        self._cached_scalar_products = []

    def detach(self) -> 'LocalDes':
        descriptor = {k: t.detach() for k, t in self.descriptor.items()}
        detached = LocalDes(descriptor,
                            species=self.species,
                            norm=self.norm.detach())
        detached._cached_scalar_products = [[t.detach() for t in wrt]
                                            for wrt in self._cached_scalar_products]
        return detached
