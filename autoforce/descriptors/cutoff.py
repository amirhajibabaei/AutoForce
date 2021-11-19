# +
import torch
from autoforce.descriptors import Descriptor
from autoforce.typeinfo import pi, float_t
from typing import Union


class Cutoff(Descriptor):
    """
    Smooth cutoff functions.

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                dij: torch.Tensor,
                cutoff: Union[float, torch.Tensor]
                ) -> torch.Tensor:
        """
        dij are distances.
        cutoff can be either a float (for all)
        or a tensor with the same length as dij.

        """

        beyond = dij > cutoff
        result = torch.where(beyond,
                             torch.zeros(1, dtype=float_t),
                             self.smooth(dij/cutoff)
                             )
        return result

    def smooth(self, dij):
        raise NotImplementedError(
            f'{self.name}: smooth method is not implemented!')


class PolynomialCut(Cutoff):
    """
    Polynomial-type smooth cutoff function.
    Degree must be greater then or equal to 2.

    """

    def __init__(self, degree: int) -> None:
        super().__init__()
        if degree < 2:
            raise RuntimeError(f'{self.name}: degree is less than 2!')
        self.degree = degree

    def smooth(self, sij: torch.Tensor) -> torch.Tensor:
        return (1-sij)**self.degree


class CosineCut(Cutoff):
    """
    Cosine-type smooth cutoff function.

    """

    def smooth(self, sij: torch.Tensor) -> torch.Tensor:
        return sij.mul(pi).cos().add(1).mul(0.5)
