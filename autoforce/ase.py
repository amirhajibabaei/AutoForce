# +
import autoforce.cfg as cfg
import autoforce.core as core
from ase.neighborlist import (wrap_positions,
                              primitive_neighbor_list as _nl)
from ase.io import read as _read
from typing import Optional, List, Any, Union
from collections import Counter
import torch
import numpy as np
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


def localslist(conf: core.Conf,
               cutoff: core.ChemPar,
               cutoff_fn: core.Function,
               subset: Optional[List[int]] = None
               ) -> (List[core.LocalEnv], Counter):
    """
    Generates a list of data.LocalEnv from data.Conf.
    It uses ASE neighborlist generator for building LCEs
    and weights wij of the neigbors are calculated using
    cutoff and cutoff_fn arguments.

    Note that no LocalEnv is generated for isolated atoms,
    thus it also returns a count of the isolated atoms.

    The "subset" arg can be used for requesting only a
    subset of local envs. Although, the full neighborlist
    still is generated. This can be optimized in near future.

    """

    # TODO: partial neighborlist if subset not None.

    # 1.
    if cutoff.keylen != 2 or not cutoff.permsym:
        raise RuntimeError('Wrong ChemPar for cutoff!')

    natoms = len(conf.numbers)
    numbers = conf.numbers.unique().tolist()
    if subset is None:
        subset = range(natoms)

    # 2. Neighborlist and shifts sij due to pbc
    i, j, sij = _nl('ijS',
                    conf.pbc,
                    conf.cell.detach().numpy(),
                    conf.positions.detach().numpy(),
                    cutoff.as_dict(numbers, float),
                    numbers=conf.numbers.numpy())

    # 3. Displacements rij
    sij = torch.from_numpy(sij)
    shifts = (sij[..., None]*conf.cell).sum(dim=1)
    rij = conf.positions[j] - conf.positions[i] + shifts
    dij = rij.norm(dim=1)
    wij = cutoff_fn(dij, cutoff(conf.numbers[i], conf.numbers[j]))

    # 4. Split localenvs; note that neighborlist is already sorted wrt "i"
    sizes = np.bincount(i, minlength=natoms).tolist()
    i = torch.from_numpy(i).split(sizes)
    j = torch.from_numpy(j).split(sizes)
    rij = rij.split(sizes)
    wij = wij.split(sizes)

    # 5. LocalEnvs
    localenvs = []
    isolated = Counter()
    for k in subset:
        if sizes[k] == 0:
            isolated[int(conf.numbers[k])] += 1
        else:
            _i = i[k][0]
            env = core.LocalEnv(_i,
                                conf.numbers[_i],
                                conf.numbers[j[k]],
                                rij[k],
                                wij[k])
            localenvs.append(env)

    # cache
    conf._cached_isolated_atoms = isolated
    conf._cached_local_envs = localenvs

    return localenvs, isolated
