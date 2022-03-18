# +
import autoforce.cfg as cfg
import autoforce.core as core
from ase.neighborlist import wrap_positions
from ase.io import read as _read
from typing import List, Any, Union
import torch
import ase


def from_atoms(atoms: ase.Atoms) -> core.Conf:
    """
    Generates a data.Conf object from a ase.Atoms object.

    """

    # 1.
    numbers = torch.from_numpy(atoms.numbers)
    wrapped = wrap_positions(atoms.positions,
                             atoms.cell,
                             atoms.pbc)
    positions = torch.from_numpy(wrapped).to(cfg.float_t)
    cell = torch.from_numpy(atoms.cell.array).to(cfg.float_t)

    # 2.
    e, f = None, None
    if atoms.calc:
        if 'energy' in atoms.calc.results:
            e = atoms.get_potential_energy()
            e = torch.tensor(e).to(cfg.float_t)
        if 'forces' in atoms.calc.results:
            f = atoms.get_forces()
            f = torch.from_numpy(f).to(cfg.float_t)
    targets = core.Targets(energy=e, forces=f)

    # 3.
    conf = core.Conf(numbers, positions, cell, atoms.pbc, targets=targets)

    return conf


def read(*args: Any, **kwargs: Any) -> Union[core.Conf, List[core.Conf]]:
    """
    Reads Atoms and converts them to Conf.
    """
    data = _read(*args, **kwargs)
    if type(data) == ase.Atoms:
        ret = from_atoms(data)
    else:
        ret = [from_atoms(x) for x in data]
    return ret
