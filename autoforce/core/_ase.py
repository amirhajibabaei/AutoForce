# +
import autoforce.cfg as cfg
from autoforce.core.data import Data, Conf, Environ
from autoforce.core.parameter import ChemPar
from autoforce.core.function import Function
from ase.neighborlist import (wrap_positions,
                              primitive_neighbor_list as _nl)
from typing import Optional, List, Any
from collections import Counter
import torch
import numpy as np
import ase


def from_atoms(atoms: ase.Atoms) -> Conf:
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
    data = Data()
    if atoms.calc:
        if 'energy' in atoms.calc.results:
            e = atoms.get_potential_energy()
            data.energy = torch.tensor(e).to(cfg.float_t)
        if 'forces' in atoms.calc.results:
            f = atoms.get_forces()
            data.forces = torch.from_numpy(f).to(cfg.float_t)

    # 3.
    conf = Conf(numbers, positions, cell, atoms.pbc, data=data)

    return conf


def environlist(conf: Conf,
                cutoff: ChemPar,
                cutoff_fn: Function,
                subset: Optional[List[int]] = None
                ) -> (List[Environ], Counter):
    """
    Generates a list of LCEs (data.Environ) from data.Conf.
    It uses ASE neighborlist generator for building LCEs
    and weights wij of the neigbors are calculated using
    cutoff and cutoff_fn arguments.

    Note that no Environ is generated for isolated atoms,
    thus it also returns a count of the isolated atoms.

    The "subset" arg can be used for requesting only a
    subset of LCEs. Although, the full neighborlist still
    is generated. This can be optimized in near future.

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

    # 4. Split environs; note that neighborlist is already sorted wrt "i"
    sizes = np.bincount(i, minlength=natoms).tolist()
    i = torch.from_numpy(i).split(sizes)
    j = torch.from_numpy(j).split(sizes)
    rij = rij.split(sizes)
    wij = wij.split(sizes)

    # 5. Environs
    environs = []
    isolated = Counter()
    for k in subset:
        if sizes[k] == 0:
            isolated[int(conf.numbers[k])] += 1
        else:
            _i = i[k][0]
            env = Environ(_i,
                          conf.numbers[_i],
                          conf.numbers[j[k]],
                          rij[k],
                          wij[k])
            environs.append(env)

    return environs, isolated
