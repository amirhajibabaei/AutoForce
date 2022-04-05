# +
import autoforce.core as core
from autoforce.functions import Overlaps
from torch import Tensor
import torch
from autoforce.aliases import Descriptor_t


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
                   ) -> Descriptor_t:
        a, b, data = self.overlaps.function(rij, numbers, wij)
        d = {(int(p), int(q)): t for p, q, t in zip(a, b, data)}
        return d
