# +
from autoforce.core.dataclasses import LocalEnv, LocalDes
from torch import Tensor
from abc import ABC, abstractmethod


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

    def __call__(self, e: LocalEnv) -> LocalDes:
        d = self.forward(e)
        d.index = int(e.index)
        if d.norm is None:
            d.norm = self.scalar_product(d, d).sqrt().view([])
        if d.species is None:
            d.species = int(e.number)
        return d

    @abstractmethod
    def forward(self, e: LocalEnv) -> LocalDes:
        ...

    @abstractmethod
    def scalar_product(self, x: LocalDes, y: LocalDes) -> Tensor:
        ...
