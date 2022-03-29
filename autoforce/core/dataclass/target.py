# +
from torch import Tensor
from typing import Optional


class Target:

    __slots__ = ('energy', 'forces')

    def __init__(self,
                 *,
                 energy: Optional[Tensor] = None,
                 forces: Optional[Tensor] = None
                 ) -> None:
        """
        Self explanatory.

        """
        self.energy = energy
        self.forces = forces
