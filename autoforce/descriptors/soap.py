# +
import autoforce.core as core
from .overlaps import Overlaps
from torch import Tensor
import torch


class SOAP(core.Descriptor):

    def __init__(self,
                 cutoff: core.Cutoff,
                 cutoff_fn: core.Cutoff_fn,
                 lmax: int,
                 nmax: int
                 ) -> None:
        super().__init__(cutoff, cutoff_fn)
        self.overlaps = Overlaps(lmax, nmax)

    def forward(self,
                number: Tensor,
                numbers: Tensor,
                rij: Tensor,
                wij: Tensor
                ) -> core.LocalDes:
        a, b, data = self.overlaps(rij, numbers, wij)
        t = torch.sparse_coo_tensor([a.tolist(), b.tolist()], data)
        d = core.LocalDes(t)
        return d

    def scalar_product(self,
                       u: core.LocalDes,
                       v: core.LocalDes
                       ) -> Tensor:
        x, = u.tensors
        y, = v.tensors
        k = torch.sparse.sum(x*y)
        return k
