# +
import torch
from torch import Tensor
from abc import ABC, abstractmethod


class Constraint(ABC):
    """
    Abstract Base Class for constraints.

    Two methods should be defined:

        1. apply method converts an unbound number
           into a value consistent with the constraint

        2. inverse method is the inverse of apply

    """

    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def inverse(self, x: Tensor) -> Tensor:
        ...


class RangeConstraint(Constraint):

    def __init__(self, a, b):

        if a > b:
            raise RuntimeError('Lower bound is larger than upper bound!')

        self.a = torch.as_tensor(a)
        self.b = torch.as_tensor(b)
        self.l = b - a

    def __repr__(self):
        return f'{self.__class__.__name__}({self.a}, {self.b})'

    def apply(self, x):
        x = torch.as_tensor(x)
        y = self.a + self.l*torch.special.expit(x)
        return y

    def inverse(self, y):
        y = torch.as_tensor(y)
        x = (y-self.a)/self.l
        return torch.special.logit(x)


def test_RangeConstraint():
    r = RangeConstraint(0., 1.)
    x = torch.tensor([0.0, 0.5, 1.0])
    test = r.apply(r.inverse(x)).allclose(x)
    return test


if __name__ == '__main__':
    a = test_RangeConstraint()
    print(a)
