# +
import theforce.cl as cline
from theforce.calculator.active import FilterDeltas
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.io import read
from ase import units
import numpy as np
import os


def relax(atoms, fmax=0.01, cell=False, mask=None, trajectory='relax.traj', rattle=0.0, confirm=False):
    """
    atoms:        ASE atoms
    fmax:         maximum forces
    cell:         if True, minimize stress
    mask:         stress components for relaxation
    trajectory:   traj file name
    rattle:       rattle atoms at initial step (recommended ~0.05)
    confirm:      if True, test DFT for the final state
    """

    calc = cline.gen_active_calc()
    atoms.rattle(rattle, rng=np.random)
    atoms.set_calculator(calc)

    # define and run relaxation dynamics
    if cell:
        filtered = ExpCellFilter(filtered, mask=mask)
    else:
        filtered = atoms
    dyn = LBFGS(filtered, trajectory=trajectory)
    dyn.run(fmax)

    # confirm:
    if confirm:
        calc._test()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Machine Learning accelerated relaxation')
    parser.add_argument('-i', '--input', default='POSCAR', type=str,
                        help='the initial coordinates of atoms, POSCAR, xyz, cif, etc.')
    parser.add_argument('-o', '--output', default='CONTCAR', type=str,
                        help='the final coordinates of atoms')
    args = parser.parse_args()
    atoms = read(args.input)
    kwargs = cline.get_default_args(relax)
    cline.update_args(kwargs)
    relax(atoms, **kwargs)
    atoms.write(args.output)
