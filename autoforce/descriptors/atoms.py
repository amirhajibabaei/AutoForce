# +
import numpy as np
import torch
import ase
from autoforce.typeinfo import float_t
from typing import Union, Optional


class AutoAtoms:
    """
    It extracts the essential information (see __slots__)
    from an ase.Atoms object and stores them in appropriate
    types for descriptor calculations:
        1) torch.Tensor instead of np.ndarray (for autograd).
        2) float32 (=float_t) instead of float64 (for memory efficiency).
    All the descriptors are essentially derived from "positions" 
    and "cell". Thus their datatype will be automatically 
    propagated through descriptor transformations.
    """

    __slots__ = [
        'numbers',
        'positions',
        'cell',
        'pbc',
        '_requires_grad',
        '_nl'
    ]

    def __init__(self, atoms: ase.Atoms,
                 requires_grad: Optional[bool] = False
                 ) -> None:
        self.numbers = torch.from_numpy(atoms.numbers)
        self.pbc = atoms.pbc
        wraped = ase.neighborlist.wrap_positions(atoms.positions,
                                                 atoms.cell,
                                                 pbc=atoms.pbc)
        # Use float32 for memory efficiency
        self.positions = torch.from_numpy(wraped).to(float_t)
        self.cell = torch.from_numpy(atoms.cell.array).to(float_t)
        self.requires_grad = requires_grad

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value
        self.positions.requires_grad = value
        self.cell.requires_grad = value

    def build_neighbor_list(self, cutoff: Union[float, dict]) -> None:
        i, j, sij = ase.neighborlist.primitive_neighbor_list('ijS',
                                                             self.pbc,
                                                             self.cell.numpy(),
                                                             self.positions.numpy(),
                                                             cutoff,
                                                             numbers=self.numbers.numpy())
        count = np.bincount(i).tolist()
        i = torch.from_numpy(i)
        j = torch.from_numpy(j)
        sij = torch.from_numpy(sij)
        rij = (self.positions[j] -
               self.positions[i] +
               (sij[..., None] * self.cell).sum(dim=1))
        self._nl = (count, i, j, sij, rij)
