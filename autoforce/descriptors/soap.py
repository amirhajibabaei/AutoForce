# +
import autoforce.core as core
from autoforce.functions import Overlaps
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

    def descriptor(self,
                   number: Tensor,
                   numbers: Tensor,
                   rij: Tensor,
                   wij: Tensor
                   ) -> core.LocalDes:
        a, b, data = self.overlaps.function(rij, numbers, wij)
        d = core.LocalDes(data, meta=(a.tolist(), b.tolist()))
        return d

    def scalar_product(self,
                       u: core.LocalDes,
                       v: core.LocalDes
                       ) -> Tensor:
        x = torch.sparse_coo_tensor(u.meta, *u.tensors)
        y = torch.sparse_coo_tensor(v.meta, *v.tensors)
        k = torch.sparse.sum(x*y)
        return k
