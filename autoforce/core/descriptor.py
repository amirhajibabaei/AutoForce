# +
from autoforce.core.dataclasses import Conf, LocalEnv, LocalDes, Basis
from autoforce.core.cutoff import Cutoff, Cutoff_fn
import torch
from torch import Tensor
from collections import defaultdict
import itertools
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class Descriptor(ABC):
    """
    A Descriptor converts a "LocalEnv" object
    into a "LocalDes" object:

        Descriptor: LocalEnv -> LocalDes

    See LocalEnv and LocalDes in core.dataclasses.


    Required methods:
        1) forward
        2) scalar_product


    The "forward" method:
    The body of a Descriptor should be implemented
    as the "forward" method. The main component of
    a LocalDes object is the "tensors" attribute.
    "meta" are arbirtary data for index-keeping,
    etc which are handled opaquely by the Descriptor.
    Other attributes are "index", "species", and "norm"
    which are automatically handled and, generally,
    should not be specified within the forward method.


    The "scalar_product" method:
    Returns the scalar product of two LocalDes objects.
    In particlar, the "norm" of a LocalDes object is
    defined as the square root of its scalar product
    by itself.

    """

    instances = 0

    def __init__(self,
                 cutoff: Cutoff,
                 cutoff_fn: Cutoff_fn
                 ) -> None:

        self.cutoff = cutoff
        self.cutoff_fn = cutoff_fn

        # Assign a global index for this instance
        self.index = Descriptor.instances
        Descriptor.instances += 1
        self.basis = tuple()

    @abstractmethod
    def forward(self,
                number: Tensor,
                numbers: Tensor,
                rij: Tensor,
                wij: Tensor
                ) -> LocalDes:
        """
        Should be implemented in a subclass.

        """
        ...

    @abstractmethod
    def scalar_product(self,
                       x: LocalDes,
                       y: LocalDes
                       ) -> Tensor:
        """
        Should be implemented in a subclass.

        """
        ...

    def get_descriptor(self, e: LocalEnv) -> LocalDes:
        while len(e._cached_descriptors) < Descriptor.instances:
            e._cached_descriptors.append(None)
        if e._cached_descriptors[self.index] is None:
            numbers, rij, dij, cij = self.cutoff.get_neighbors(e)
            wij = self.cutoff_fn(dij, cij)
            d = self.forward(e.number, numbers, rij, wij)
            d.index = int(e.index)
            if d.norm is None:
                d.norm = self.scalar_product(d, d).sqrt().view([])
            if d.species is None:
                d.species = int(e.number)
            e._cached_descriptors[self.index] = d
        return e._cached_descriptors[self.index]

    def get_descriptors(self, conf: Conf) -> List[LocalDes]:
        if conf._cached_local_envs is None:
            raise RuntimeError(f'{conf._cached_local_envs = }')
        return [self.get_descriptor(l) for l in conf._cached_local_envs]

    def get_scalar_products(self,
                            d: LocalDes,
                            basis: Basis
                            ) -> List[Tensor]:
        # 1. update cache: d._cached_scalar_products
        while len(d._cached_scalar_products) <= basis.index:
            d._cached_scalar_products.append([])
        m = len(d._cached_scalar_products[basis.index])
        new = [self.scalar_product(base, d) if active else None
               for base, active in zip(basis.descriptors[d.species][m:],
                                       basis.active[d.species][m:])]
        d._cached_scalar_products[basis.index].extend(new)

        # 2. retrieve from cache
        out = itertools.compress(d._cached_scalar_products[basis.index],
                                 basis.active[d.species])
        return list(out)

    def get_scalar_products_dict(self,
                                 conf: Conf,
                                 basis: Basis
                                 ) -> (Dict, Dict):
        prod = defaultdict(list)
        norms = defaultdict(list)
        for d in self.get_descriptors(conf):
            k = self.get_scalar_products(d, basis)
            prod[d.species].append(k)
            norms[d.species].append(d.norm)
        return prod, norms

    def get_gram_dict(self, basis: Basis) -> Dict:
        gram = {}
        for species, descriptors in basis.descriptors.items():
            z = itertools.compress(descriptors, basis.active[species])
            gram[species] = torch.stack(
                [torch.stack(self.get_scalar_products(b, basis)) for b in z]
            )
        return gram

    def new_basis(self) -> int:
        new = Basis()
        new.index = len(self.basis)
        self.basis = (*self.basis, new)
        return new
