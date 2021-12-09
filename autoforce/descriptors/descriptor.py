# +
import autoforce.cfg as cfg
from autoforce.parameters import Chemsor
from autoforce.descriptors import Configuration, ChemEnv, Cutoff
import torch
from torch import Tensor
from ase import Atoms
from typing import List, Callable, Optional, Any
from abc import ABC, abstractmethod


class Vector:

    __slots__ = ('data', 'indices', '_species', '_norm')

    def __init__(self, data: Tensor, indices: Optional[Any] = None):
        self.data = data
        self.indices = indices
        self._species = None
        self._norm = None

    def detach(self):
        v = Vector(self.data.detach(), self.indices)
        v._species = self._species
        v._norm = self._norm.detach()
        return v

    def clone(self):
        v = Vector(self.data.clone(), self.indices)
        v._species = self._species
        v._norm = self._norm.clone()
        return v

    def __repr__(self):
        return f'Vector(species={self._species})'


class Descriptor(ABC):

    def __init__(self, cutoff_fn: Cutoff) -> None:
        self.cutoff_fn = cutoff_fn

    def __call__(self,
                 conf: Configuration,
                 cutoff: Chemsor,
                 scale: Optional[Chemsor] = None,
                 subset: Optional[List[int]] = None
                 ) -> List[Vector]:

        chemical_envs = ChemEnv.from_configuration(conf,
                                                   cutoff,
                                                   subset=subset)
        descriptors = []

        for x in chemical_envs:

            def mapping(rij):
                dij = rij.norm(dim=1)
                rc = cutoff(x.number, x.numbers)
                weights = self.cutoff_fn(dij, rc)
                if scale is not None:
                    sij = scale(x.number, x.numbers).view(-1, 1)
                    rij = rij / sij
                v = self.vector(rij, x.numbers, weights)
                return v

            if x.is_isolated:
                y = Vector(None)
            else:
                y = mapping(x.rij)

            y._species = x.number
            y._norm = self.product([y], [y]).sqrt().view([])
            descriptors.append(y)

        return descriptors

    def from_atoms(self,
                   atoms: Atoms,
                   cutoff: Chemsor,
                   scale: Optional[Chemsor] = None,
                   subset: Optional[List[int]] = None,
                   requires_grad: Optional[bool] = False
                   ) -> List[Vector]:

        conf = Configuration.from_atoms(atoms, requires_grad=requires_grad)
        descriptors = self(conf, cutoff, scale=scale, subset=subset)

        return conf, descriptors

    def product(self, X: List[Vector], Y: List[Vector]) -> Tensor:
        prod = []
        for x in X:
            if x.data is None:
                prod.append(torch.zeros(len(Y), dtype=cfg.float_t))
                continue
            row = []
            for y in Y:
                if y.data is None or x._species != y._species:
                    z = cfg.zero
                else:
                    z = self.vector_product(x, y)
                row.append(z)
            prod.append(torch.stack(row))
        return torch.stack(prod)

    @abstractmethod
    def vector(self, rij: Tensor, numbers: Tensor, weights: Tensor) -> Vector:
        ...

    @abstractmethod
    def vector_product(self, u: Vector, v: Vector) -> Tensor:
        ...
