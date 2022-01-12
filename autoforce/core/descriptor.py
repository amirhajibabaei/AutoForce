# +
from autoforce.core.dataclasses import Conf, LocalEnv, LocalDes
import torch
from torch import Tensor
from collections import defaultdict
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


    TODO:
        1. Explain "basis" attribute.
        2. Explain "orientation" method.
        3. Explain "active_set" attribute.

    """

    def __init__(self) -> None:
        self.basis = defaultdict(list)
        self.active_set = defaultdict(list)

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

    def __call__(self, e: LocalEnv) -> LocalDes:
        d = self.forward(e)
        d.index = int(e.index)
        if d.norm is None:
            d.norm = self.scalar_product(d, d).sqrt().view([])
        if d.species is None:
            d.species = int(e.number)
        return d

    def append(self, d: LocalDes) -> None:
        self.basis[d.species].append(d.detach())
        self.active_set[d.species].append(True)

    def get_orientation(self, d: LocalDes) -> List[Tensor]:
        # update cache: d._cached_orientation
        basis = self.basis[d.species]
        active = self.active_set[d.species]
        m = len(d._cached_orientation)
        for b, a in zip(basis[m:], active[m:]):
            if a:
                cosine = self.scalar_product(b, d)/(b.norm*d.norm)
            else:
                cosine = None
            d._cached_orientation.append(cosine)
        # retrieve from cache
        z = zip(d._cached_orientation, active)
        out = [cosine for cosine, a in z if a]
        return out

    def get_gram_matrix(self) -> Dict:
        out = {}
        for species, basis in self.basis.items():
            z = zip(basis, self.active_set[species])
            out[species] = torch.stack(
                [torch.stack(self.get_orientation(b)) for b, a in z if a]
            )
        return out

    def get_descriptors(self, conf: Conf) -> List:
        if conf._cached_LocalDes is None:
            if conf._cached_LocalEnv is None:
                raise RuntimeError('no LocalENV list!')
            conf._cached_LocalDes = [self(l) for l in conf._cached_LocalEnv]
        return conf._cached_LocalDes

    def get_orientations(self, conf: Conf) -> Dict:
        out = defaultdict(list)
        for d in self.get_descriptors(conf):
            k = self.get_orientation(d)
            out[d.species].append(k)
        return out
