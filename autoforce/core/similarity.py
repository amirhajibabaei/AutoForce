# +
from autoforce.core.dataclasses import LocalEnv, LocalDes
from torch import Tensor
from abc import ABC, abstractmethod


class Similarity(ABC):
    """
    Similarity is a combination of the "descriptor"
    and "kernel" functions.
    The descriptor function converts an "LocalEnv"
    object into a "LocalDes" object.
    The kernel return a scalar value, given two
    LocalDes object.

    """

    @abstractmethod
    def descriptor(self, e: LocalEnv) -> LocalDes:
        ...

    @abstractmethod
    def kernel(self, x: LocalDes, y: LocalDes) -> Tensor:
        ...
