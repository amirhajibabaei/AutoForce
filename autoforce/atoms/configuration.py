# +
import numpy as np
import torch
import ase
from ase.neighborlist import wrap_positions
from autoforce.typeinfo import float_t
from typing import Union, Optional


class Configuration:
    """
    A class for atomic configurations.
    A configuration is defined by:

        * atomic numbers
        * atomic positions
        * cell
        * pbc

    This class holds these information in formats
    appropriate for descriptor transformations:

        * the positions will be wrapped
        * uses torch.Tensor instead of np.ndarray
        * uses float_t (=float_32) instead of float64

    The torch.Tensor class is used for enabling
    automatic differenstiation.

    The positions are wrapped because, although
    modulo-cell-translations with pbc=True yield
    equivalent atomic configurations, they affect
    gradients wrt cell calculated by automatic
    differentiation.

    Using float32 can increase the memory efficiency
    since the descriptors and their gradients are
    generally high dimensional and consume a lot of
    memory. All descriptors are essentially derived
    from "positions" and "cell". Thus their datatype
    will be automatically propagated through descriptor
    transformations.

    """

    __slots__ = [
        'numbers',
        'positions',
        'cell',
        'pbc',
        '_requires_grad',
    ]

    def __init__(self,
                 numbers: np.ndarray,
                 positions: np.ndarray,
                 cell: np.ndarray,
                 pbc: np.ndarray,
                 requires_grad: Optional[bool] = False
                 ) -> None:
        """
        * numbers: atomic numbers (:)

        * positions:  atomic positions (:, 3)

        * cell: simulation cell (3, 3)

        * pbc: periodic boundary conditions (3)

        * requires_grad: attribute of positions and cell

        """

        self.numbers = torch.from_numpy(numbers)
        wrapped = wrap_positions(positions, cell, pbc)
        self.positions = torch.from_numpy(wrapped).to(float_t)
        self.cell = torch.from_numpy(cell).to(float_t)
        self.pbc = pbc
        self.requires_grad = requires_grad

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value
        self.positions.requires_grad = value
        self.cell.requires_grad = value

    def __repr__(self) -> str:
        z, c = np.unique(self.numbers, return_counts=True)
        counts = ', '.join([f'{a}: {b}' for a, b in zip(z, c)])
        return f'Configuration({{{counts}}})'

    @staticmethod
    def from_atoms(atoms: ase.Atoms,
                   requires_grad: Optional[bool] = False
                   ):
        """
        Generates a Configuration from an ase.Atoms object.

        """

        return Configuration(atoms.numbers,
                             atoms.positions,
                             atoms.cell.array,
                             atoms.pbc,
                             requires_grad=requires_grad
                             )

    def as_atoms(self) -> ase.Atoms:

        atoms = ase.Atoms(numbers=_numpy(self.numbers),
                          positions=_numpy(self.positions),
                          cell=_numpy(self.cell),
                          pbc=self.pbc)

        return atoms


def _numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().numpy()


def test_Configuration_from_atoms():
    """
    Test if from_atoms works.

    """

    from ase.build import bulk

    atoms = bulk('Au').repeat(3)
    conf = Configuration.from_atoms(atoms, requires_grad=True)
    conf.as_atoms()

    return True


if __name__ == '__main__':
    test_Configuration_from_atoms()
