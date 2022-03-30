# +
from .function import Function
from torch import Tensor
from abc import abstractmethod


class Bijection(Function):
    """
    A Bijection is a Function which, in addition
    to the "forward" method, has an "inverse"
    method such that

        x = inverse(forward(x))

    """

    @abstractmethod
    def inverse(self, x: Tensor) -> Tensor:
        ...
