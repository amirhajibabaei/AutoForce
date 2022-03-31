# +
from .function import Function
from torch import Tensor
from abc import abstractmethod


class Bijection(Function):
    """
    A "Bijection" is a "Function" which, in
    addition to the "function" method, has
    the "inverse" method such that

        x = inverse(function(x))

    """
    @abstractmethod
    def function(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def inverse(self, x: Tensor) -> Tensor:
        ...
