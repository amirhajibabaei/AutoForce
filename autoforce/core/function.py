# +
from abc import ABC, abstractmethod
from typing import Any


class Function(ABC):
    """
    Functions are classes whose main inputs and
    outputs are "torch.Tensor" objects (rather
    than other higher level structures defined
    in this package).
    They should be viewed as simple extensions
    of python functions with the ability to be
    configured to store some static data for
    faster runtime execution.

    The main purpose of this class is to seperate
    high level abstractions, which may change
    frequently, from the actual implementation.

    Thus, the key restriction is that they shouldn't
    contain any parameters that may change during
    execution (i.e. hyperparameters).
    Also they should not store any Tensor for which
    requires_grad = True.

    The main body should be implemented as the
    "forward" method imilar to "torch.autograd.Function".
    Except, forward method in the latter is restricted
    to "@staticmethod" while here it is a normal method.
    This similarity is enforced in anticipation of
    custom gradients using the "backward" method
    in some cases for faster execution.

    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ...
