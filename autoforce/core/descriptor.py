# +
from autoforce.core.dataclasses import Conf, LocalEnv, LocalDes
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

    def __init__(self, *basis: Optional[Any]) -> None:
        """
        args:    a tuple of "Basis" objects.

        """
        # Assign a global index for this instance
        self.index = Descriptor.instances
        Descriptor.instances += 1

        # Assign an index for input basis
        for i, b in enumerate(basis):
            b.index = i
        self.basis = basis

        # If no input basis, make one!
        if len(self.basis) == 0:
            self.new_basis()

    @abstractmethod
    def forward(self, e: LocalEnv) -> LocalDes:
        """
        Should be implemented in a subclass.
        """
        ...

    @abstractmethod
    def scalar_product(self, x: LocalDes, y: LocalDes) -> Tensor:
        """
        Should be implemented in a subclass.
        """
        ...

    def _forward(self, e: LocalEnv) -> LocalDes:
        while len(e._cached_descriptors) < Descriptor.instances:
            e._cached_descriptors.append(None)
        if e._cached_descriptors[self.index] is None:
            d = self.forward(e)
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
        return [self._forward(l) for l in conf._cached_local_envs]

    def _get_scalar_products(self,
                             d: LocalDes,
                             wrt: Optional[int] = 0
                             ) -> List[Tensor]:
        # 1. update cache: d._cached_scalar_products
        while len(d._cached_scalar_products) <= wrt:
            d._cached_scalar_products.append([])
        basis = self.basis[wrt].descriptors[d.species]
        active = self.basis[wrt].active[d.species]
        m = len(d._cached_scalar_products[wrt])
        for b, a in zip(basis[m:], active[m:]):
            if a:
                prod = self.scalar_product(b, d)
            else:
                prod = None
            d._cached_scalar_products[wrt].append(prod)
        # 2. retrieve from cache
        # slow: 4.3e-5 sec for len=1000
        #y = [p for p, a in zip(d._cached_scalar_products, active) if a]
        # faster: 2.3e-5 sec for len=1000
        y = itertools.compress(d._cached_scalar_products[wrt], active)
        return list(y)

    def get_scalar_products(self,
                            conf: Conf,
                            wrt: Optional[int] = 0
                            ) -> (Dict, Dict):
        prod = defaultdict(list)
        norms = defaultdict(list)
        for d in self.get_descriptors(conf):
            k = self._get_scalar_products(d, wrt=wrt)
            prod[d.species].append(k)
            norms[d.species].append(d.norm)
        return prod, norms

    def get_gram_matrix(self, wrt: Optional[int] = 0) -> Dict:
        gram = {}
        for species, basis in self.basis[wrt].descriptors.items():
            z = itertools.compress(basis, self.basis[wrt].active[species])
            gram[species] = torch.stack(
                [torch.stack(self._get_scalar_products(b, wrt=wrt)) for b in z]
            )
        return gram

    def new_basis(self) -> int:
        new = Basis()
        new.index = len(self.basis)
        self.basis = (*self.basis, new)
        return new.index


class Basis:

    def __init__(self):
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
