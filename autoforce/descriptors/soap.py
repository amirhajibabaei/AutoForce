# +
from autoforce.core import Descriptor, LocalEnv, LocalDes
from autoforce.descriptors.overlaps import Overlaps
from torch import Tensor
import torch


class SOAP(Descriptor):

    def __init__(self, lmax: int, nmax: int):
        super().__init__()
        self.overlaps = Overlaps(lmax, nmax)

    def forward(self, e: LocalEnv) -> LocalDes:
        a, b, data = self.overlaps(e.rij, e.numbers, e.wij)
        t = torch.sparse_coo_tensor([a.tolist(), b.tolist()], data)
        d = LocalDes(t)
        return d

    def scalar_product(self, u: LocalDes, v: LocalDes) -> Tensor:
        x, = u.tensors
        y, = v.tensors
        k = torch.sparse.sum(x*y)
        return k
