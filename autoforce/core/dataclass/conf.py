# +
from .target import Target
from torch import Tensor
from typing import List, Optional


class Conf:
    """
    An atomic configuration.

    Same role as ase.Atoms, except the data
    are stored as torch.Tensor instead of
    np.ndarray, for using the autograd.


    Keywords:
    numbers      atomic numbers
    positions    self explanatory
    cell         self explanatory
    pbc          periodic boundary conditions
    target       energy & forces

    """

    __slots__ = ('numbers',
                 'positions',
                 'cell',
                 'pbc',
                 'target',
                 'number_of_atoms',
                 'unique_counts',
                 '_cached_local_envs',
                 '_cached_isolated_atoms')

    def __init__(self,
                 numbers: Tensor,
                 positions: Tensor,
                 cell: Tensor,
                 pbc: List[bool],
                 target: Optional[Target] = None
                 ) -> None:
        """
        Self explanatory.

        """

        if positions.requires_grad != cell.requires_grad:
            raise RuntimeError('requires_grad should be the '
                               'same for positions and cell!'
                               )

        self.numbers = numbers
        self.number_of_atoms = int(numbers.numel())
        u, c = numbers.unique(return_counts=True)
        self.unique_counts = {int(a): int(b) for a, b in zip(u, c)}
        self.positions = positions
        self.cell = cell
        self.pbc = pbc

        if target is None:
            target = Target()
        self.target = target

        # cache
        self._cached_local_envs = None

    @property
    def requires_grad(self) -> bool:
        return self.positions.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self.positions.requires_grad = value
        self.cell.requires_grad = value
