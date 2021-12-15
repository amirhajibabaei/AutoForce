# +
import autoforce.cfg as cfg
from autoforce.core.function import Function
import torch
from torch import Tensor
from abc import abstractmethod
from typing import Optional


class Cutoff(Function):
    """
    Smooth cutoff functions.

    """

    def forward(self, dij: Tensor, cutoff: Tensor) -> Tensor:
        """
        dij        distances
        cutoff     can be either a scalar or a tensor
                   with the same length as dij
        """

        beyond = dij > cutoff
        result = torch.where(beyond,
                             cfg.zero,
                             self.smooth(dij/cutoff)
                             )
        return result

    @abstractmethod
    def smooth(self, dij: Tensor) -> Tensor:
        ...


class PolynomialCut(Cutoff):
    """
    Polynomial-type smooth cutoff function.
    Degree must be greater then or equal to 2.

    """

    def __init__(self, degree: Optional[int] = 2) -> None:
        super().__init__()
        if degree < 2:
            raise RuntimeError('PolynomialCut: degree is less than 2!')
        self.degree = degree

    def smooth(self, sij: torch.Tensor) -> torch.Tensor:
        return (1-sij)**self.degree


class CosineCut(Cutoff):
    """
    Cosine-type smooth cutoff function.

    """

    def smooth(self, sij: torch.Tensor) -> torch.Tensor:
        return sij.mul(cfg.pi).cos().add(1).mul(0.5)
