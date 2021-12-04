# +
import numpy as np
import torch
import ase
from ase.neighborlist import primitive_neighbor_list
from autoforce.atoms import Configuration, Chemsor
from typing import Union, Optional


class ChemEnv:
    """
    A class for local chemical environments (LCEs).
    A ChemEnv is defined by

        * number: species of the central atom
        * numbers: species of the neighboring atoms
        * rij: coordinates of the neighboring atoms 
               relative to the central atom

    Since many instances of this class are expected
    to be created, we use __slots__ for efficiency.

    If the central atom is isolated (with a given
    cutoff), then:

        numbers = rij = None

    """

    __slots__ = ['number', 'numbers', 'rij']

    def __init__(self,
                 number: torch.Tensor,
                 numbers: Optional[torch.Tensor] = None,
                 rij: Optional[torch.Tensor] = None):
        self.number = number
        self.numbers = numbers
        self.rij = rij

    @property
    def is_isolated(self) -> bool:
        return self.numbers is None

    def clone(self):  # -> ChemEnv
        return ChemEnv(self.number, self.numbers, self.rij.clone())

    def detach(self):  # -> ChemEnv
        return ChemEnv(self.number, self.numbers, self.rij.detach())

    @property
    def requires_grad(self) -> bool:
        return self.rij.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self.rij.requires_grad = value

    def __repr__(self) -> str:
        if self.is_isolated:
            counts = ''
        else:
            z, c = np.unique(self.numbers, return_counts=True)
            counts = ', '.join([f'{a}: {b}' for a, b in zip(z, c)])
        return f'ChemEnv({int(self.number)} -> {{{counts}}})'

    @staticmethod
    def from_atoms(atoms: ase.Atoms,
                   cutoff: Chemsor,
                   subset: Optional[list] = None,
                   requires_grad: Optional[bool] = False
                   ) -> (Configuration, list):
        """
        It generates Configuration and a list of ChemEnv
        objects (see from_configuration).

        It returns both the Configuration and the env list.

        """

        conf = Configuration.from_atoms(atoms, requires_grad=requires_grad)
        envs = ChemEnv.from_configuration(conf, cutoff, subset=subset)

        return conf, envs

    @staticmethod
    def from_configuration(conf: Configuration,
                           cutoff: Chemsor,
                           subset: Optional[list] = None
                           ) -> list:
        """
        It returns a list of ChemEnv objects; one for
        each atom in the atomic configuration.
        The list is ordered and for isolated atoms

            numbers = rij = None.

        If "subset" is given as a list of indices, then
        only the envs for the given indices will be
        returned.

        In future further optimization can be made if
        only a subset of environments is needed. Currently
        the entire neighborlist is generated and then
        truncated.

        """

        if cutoff.index_dim != 2 or not cutoff.perm_sym:
            raise RuntimeError('Wrong Chemsor for cutoff!')
        cutoff = cutoff.as_dict(conf.species(), float)

        # 1. Neighborlist and shifts sij due to pbc
        i, j, sij = primitive_neighbor_list('ijS',
                                            conf.pbc,
                                            _numpy(conf.cell),
                                            _numpy(conf.positions),
                                            cutoff,
                                            numbers=_numpy(conf.numbers))

        # 2. Displacements rij
        sij = torch.from_numpy(sij)
        shifts = (sij[..., None]*conf.cell).sum(dim=1)
        rij = conf.positions[j] - conf.positions[i] + shifts

        # 3. Split Envs; note that neighborlist is already sorted wrt "i"
        sizes = np.bincount(i).tolist()
        z = zip(torch.from_numpy(i).split(sizes),
                torch.from_numpy(j).split(sizes),
                rij.split(sizes))

        # 4. Create Envs; None for isolated atoms
        envs = len(conf.numbers)*[None]
        for _i, j, rij in z:
            i = _i[0]
            envs[i] = ChemEnv(conf.numbers[i], conf.numbers[j], rij)

        # 5. Envs for isolated atoms
        for i, env in enumerate(envs):
            if env is None:
                envs[i] = ChemEnv(conf.numbers[i])

        # 6. Truncate if only a subset is needed
        if subset is not None:
            envs = [envs[i] for i in subset]

        return envs


def _numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().numpy()


def test_ChemEnv():
    """
    Test if ChemEnv works without runtime errors.

    """

    from ase.build import bulk

    atoms = bulk('Au').repeat(3)

    cutoff = Chemsor({(79, 79): 6.})
    conf = Configuration.from_atoms(atoms, requires_grad=True)
    envs = ChemEnv.from_configuration(conf, cutoff)

    conf, envs = ChemEnv.from_atoms(atoms, cutoff, requires_grad=True)

    envs[0].clone()

    envs[0].detach()

    return True


def test_ChemEnv_isolated_atoms():
    """
    Test if ChemEnv treats isolated atoms correctly.

    """

    from ase.build import bulk

    atoms = bulk('Au', cubic=True).repeat(2)
    atoms.pbc = False
    atoms[-1].position += 100.

    cutoff = Chemsor({(79, 79): 6.})

    conf, [env] = ChemEnv.from_atoms(atoms, cutoff, subset=[-1])

    return env.is_isolated


if __name__ == '__main__':
    test_ChemEnv()
    test_ChemEnv_isolated_atoms()
