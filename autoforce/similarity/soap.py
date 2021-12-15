# +
from autoforce.core import Similarity, Descriptor, Environ
from autoforce.similarity.overlaps import Overlaps
from torch import Tensor
import torch


class SOAP(Similarity):

    def __init__(self, lmax: int, nmax: int):
        self.overlaps = Overlaps(lmax, nmax)

    def descriptor(self, e: Environ) -> Descriptor:
        a, b, data = self.overlaps(e.rij, e.numbers, e.wij)
        t = torch.sparse_coo_tensor([a.tolist(), b.tolist()], data)
        d = Descriptor(t)
        d.index = int(e.index)
        d.species = int(e.number)
        d.norm = self.kernel(d, d).sqrt()
        return d

    def kernel(self, u: Descriptor, v: Descriptor) -> Tensor:
        x, = u.tensors
        y, = v.tensors
        k = torch.sparse.sum(x*y)
        return k
