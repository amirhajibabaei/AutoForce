# +
import autoforce.cfg as cfg
from .dataclasses import LocalEnv
from .parameter import ChemPar
from .function import Function
import torch
from torch import Tensor
from abc import abstractmethod
from typing import Optional, Dict


class Cutoff(ChemPar):

    def __init__(self,
                 default: float,
                 values: Optional[Dict[int, float]] = None
                 ) -> None:
        super().__init__(default=default,
                         values=values,
                         clone=True,
                         keylen=2,
                         permsym=True,
                         bijection=None,
                         requires_grad=False
                         )

    def get_neighbors(self, e: LocalEnv) -> (Tensor, Tensor, Tensor, Tensor):
        cutoff = self(e.number, e.numbers)
        m = e.dij < cutoff
        return e.numbers[m], e.rij[m], e.dij[m], cutoff[m]


class Cutoff_fn(Function):
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


class PolynomialCut(Cutoff_fn):
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


class CosineCut(Cutoff_fn):
    """
    Cosine-type smooth cutoff function.

    """

    def smooth(self, sij: torch.Tensor) -> torch.Tensor:
        return sij.mul(cfg.pi).cos().add(1).mul(0.5)
