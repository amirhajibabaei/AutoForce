# +
import autoforce.core as core
from torch import Tensor


class DotProductKernel(core.Kernel):

    def __init__(self,
                 descriptor: core.Descriptor,
                 exponent: core.ChemPar
                 ) -> None:
        super().__init__(descriptor, exponent)

    def forward(self,
                uv: Tensor,
                u: Tensor,
                v: Tensor
                ) -> Tensor:
        return uv/(u*v)
