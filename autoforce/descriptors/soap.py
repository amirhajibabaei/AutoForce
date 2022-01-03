# +
import autoforce.core as core
from autoforce.descriptors.overlaps import Overlaps
from torch import Tensor
import torch


class SOAP(core.Descriptor):

    def __init__(self, lmax: int, nmax: int):
        self.overlaps = Overlaps(lmax, nmax)

    def forward(self, e: core.LocalEnv) -> core.LocalDes:
        a, b, data = self.overlaps(e.rij, e.numbers, e.wij)
        t = torch.sparse_coo_tensor([a.tolist(), b.tolist()], data)
        d = core.LocalDes(t)
        return d

    def scalar_product(self, u: core.LocalDes, v: core.LocalDes) -> Tensor:
        x, = u.tensors
        y, = v.tensors
        k = torch.sparse.sum(x*y)
        return k
