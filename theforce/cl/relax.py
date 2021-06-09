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
    confirm:      if True, do ab initio for the last step
    """

    calc = cline.gen_active_calc()
    load1 = calc.size[0]
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

    load2 = calc.size[0]

    # confirm:
    if calc.active and confirm:

        while True:
            load2 += 1
            if calc.update_data(try_fake=False):
                calc.results.clear()
                dyn.initialize()
                dyn.run(fmax=fmax)
            else:
                break

        ML = ('ML', calc.results['energy'], calc.results['forces'])
        Ab = ('Ab initio', *calc._test())
        for method, energy, forces in [ML, Ab]:
            f_rms = np.sqrt(np.mean(forces**2))
            f_max = abs(forces).max()
            report = f"""
                relaxation result ({method}):
                energy:      {energy}
                force (rms): {f_rms}
                force (max): {f_max}
            """
            if master:
                print(report)

    if master:
        print(f'\tTotal number of Ab initio calculations: {load2-load1}\n')


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
    try:
        atoms.write(args.output)
    except:
        import warnings
        alt = 'active_optimized.xyz'
        msg = f'writing to {args.output} failed -> wrote {alt}'
        warnings.warn(msg)
        atoms.write(alt, format='extxyz')
