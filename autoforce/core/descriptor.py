# +
from autoforce.core.dataclasses import Conf, LocalEnv, LocalDes
import torch
from torch import Tensor
from collections import defaultdict
import itertools
from abc import ABC, abstractmethod
from typing import Dict, List


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

    def __init__(self) -> None:
        # Assign a global index for this instance
        self.index = Descriptor.instances
        Descriptor.instances += 1

        # Self explanatory
        self.basis = Basis()

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

    def get_descriptors(self, conf: Conf) -> List:
        if conf._cached_local_envs is None:
            raise RuntimeError(f'{conf._cached_local_envs = }')
        return [self._forward(l) for l in conf._cached_local_envs]

    def _get_scalar_products(self, d: LocalDes) -> List[Tensor]:
        # update cache: d._cached_scalar_products
        basis = self.basis.descriptors[d.species]
        active = self.basis.active[d.species]
        m = len(d._cached_scalar_products)
        for b, a in zip(basis[m:], active[m:]):
            if a:
                prod = self.scalar_product(b, d)
            else:
                prod = None
            d._cached_scalar_products.append(prod)
        # retrieve from cache
        # slow: 4.3e-5 sec for len=1000
        #y = [p for p, a in zip(d._cached_scalar_products, active) if a]
        # faster: 2.3e-5 sec for len=1000
        y = list(itertools.compress(d._cached_scalar_products, active))
        return y

    def get_scalar_products(self, conf: Conf) -> Dict:
        prod = defaultdict(list)
        norms = defaultdict(list)
        for d in self.get_descriptors(conf):
            k = self._get_scalar_products(d)
            prod[d.species].append(k)
            norms[d.species].append(d.norm)
        return prod, norms

    def get_gram_matrix(self) -> Dict:
        out = {}
        for species, basis in self.basis.descriptors.items():
            z = itertools.compress(basis, self.basis.active[species])
            out[species] = torch.stack(
                [torch.stack(self._get_scalar_products(b)) for b in z]
            )
        return out


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
