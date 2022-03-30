# +
import autoforce.cfg as cfg
from autoforce.core import Cutoff_fn
from torch import Tensor
from typing import Optional


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

    def smooth(self, sij: Tensor) -> Tensor:
        return (1-sij)**self.degree


class CosineCut(Cutoff_fn):
    """
    Cosine-type smooth cutoff function.

    """

    def smooth(self, sij: Tensor) -> Tensor:
        return sij.mul(cfg.pi).cos().add(1).mul(0.5)
