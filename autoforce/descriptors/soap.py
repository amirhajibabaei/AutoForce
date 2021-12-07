# +
import torch
from torch import Tensor
from autoforce.descriptors import Descriptor, Vector, Cutoff, Overlaps


class SOAP(Descriptor):

    def __init__(self,
                 cutoff_fn: Cutoff,
                 lmax: int,
                 nmax: int
                 ) -> None:

        super().__init__(cutoff_fn)
        self.overlaps = Overlaps(lmax, nmax)

    def vector(self,
               rij: Tensor,
               numbers: Tensor,
               weights: Tensor
               ) -> Vector:

        a, b, data = self.overlaps(rij, numbers, weights)
        return Vector(data, indices=(a.tolist(), b.tolist()))

    def vector_product(self,
                       u: Vector,
                       v: Vector
                       ) -> Tensor:

        x = torch.sparse_coo_tensor(u.indices, u.data)
        y = torch.sparse_coo_tensor(v.indices, v.data)

        return torch.sparse.sum(x*y)
