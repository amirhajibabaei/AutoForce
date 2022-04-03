# +
from ..dataclass import Conf, Target
from ..parameter import Cutoff
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Sequence, Union, Any


class Regressor(ABC):
    """
    TODO:

    """

    @property
    @abstractmethod
    def cutoff(self) -> Union[Cutoff, None]:
        """
        TODO:

        """
        ...

    @abstractmethod
    def get_design_matrix(self, confs: Sequence[Conf]) -> (Tensor, Tensor, Any):
        """
        TODO:

        """
        ...

    @abstractmethod
    def set_weights(self, weights: Tensor, sections: Any) -> None:
        """
        TODO:

        """
        ...

    @abstractmethod
    def get_target(self, conf: Conf) -> Target:
        """
        TODO:

        """
        ...
