# +
from autoforce.core.data import Environ, Descriptor
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Optional, List


class Similarity(ABC):
    """
    Similarity is a combination of the "descriptor"
    and "kernel" functions.
    The descriptor function converts an "Environ"
    object into a "Descriptor" object.
    The kernel return a scalar value, given two
    Descriptor object.

    """

    @abstractmethod
    def descriptor(self, e: Environ) -> Descriptor:
        ...

    @abstractmethod
    def kernel(self, x: Descriptor, y: Descriptor) -> Tensor:
        ...
