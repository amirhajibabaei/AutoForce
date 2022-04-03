# +
import autoforce.core as core
from torch import Tensor


class DotProductKernel(core.Kernel):

    def function(self,
                 uv: Tensor,
                 u: Tensor,
                 v: Tensor
                 ) -> Tensor:
        return uv/(u*v)
