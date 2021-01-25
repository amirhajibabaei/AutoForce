# +
import theforce.cl as cline
from theforce.calculator.active import FilterDeltas
from ase import optimize
from ase.constraints import UnitCellFilter
from ase.io import read
from ase import units
import numpy as np
import os


def relax(atoms, fmax=0.01, cell=False, mask=None, algo='LBFGS', trajectory='relax.traj', rattle=0.02, confirm=True):
    """
    atoms:        ASE atoms
    fmax:         maximum forces
    cell:         if True, minimize stress
    mask:         stress components for relaxation
    algo:         algo from ase.optimize
    trajectory:   traj file name
    rattle:       rattle atoms at initial step
    confirm:      if True, test DFT for the final state
    """

    calc = cline.gen_active_calc()
    master = calc.rank == 0
    atoms.rattle(rattle, rng=np.random)
    atoms.set_calculator(calc)

    # define and run relaxation dynamics
    if cell:
        filtered = UnitCellFilter(atoms, mask=mask)
    else:
        filtered = atoms
    Min = getattr(optimize, algo)
    dyn = Min(filtered, trajectory=trajectory, master=master)
    for _ in dyn.irun(fmax):
        if calc.updated:
            dyn.initialize()

    # confirm:
    if calc.active and confirm:
        e, f = calc._test()
        err = abs(f).max()
        if err > fmax:
            u = calc.update_data(try_fake=False)
            if u > 0:
                calc.log('relax: ML model is updated at the last step!')
            else:
                calc.log('relax: desired accuracy could not be reached')
                calc.log('relax: try reducing ediff and then fdiff')
            calc.log(f"relax: run again by setting covariance = '{calc.pckl}' in ARGS")
        else:
            calc.log('relax: truly converged!')
        calc.log(f'relax: exact fmax: {err}')


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
