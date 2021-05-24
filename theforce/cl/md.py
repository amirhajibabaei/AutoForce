# +
import theforce.cl as cline
from theforce.calculator.active import FilterDeltas
from theforce.util.aseutil import init_velocities, make_cell_upper_triangular
from ase.md.npt import NPT
from ase.md.langevin import Langevin
from ase.io import read
from ase import units
import numpy as np
import os


def md(atoms, dynamics='NPT', dt=None, tem=300., picos=100, bulk_modulus=None, stress=0., mask=None,
       trajectory='md.traj', loginterval=1, append=False, rattle=0.0, tdamp=25, pdamp=100, friction=1e-3,
       ml_filter=0.8):
    """
    atoms:        ASE atoms
    dynamics:     'NPT'
    dt:           time-step in fs
    tem:          temperature in Kelvin
    picos:        pico-seconds for md
    bulk_modulus: bulk_modulus for NPT simulations. if None, NVT is performed
    stress:       external stress (GPa) for NPT
    mask:         see ase.npt.NPT
    trajectory:   traj file name
    loginterval:  for traj file
    append:       append to traj file
    rattle:       rattle atoms at initial step (recommended ~0.05)
    tdamp:        temperature damping time (fs)
    pdamp:        pressure damping time (fs)
    frinction:    for Langevin dynamics
    ml_filter:    filters force discontinuities due to ML updates range(0, 1)
    """

    calc = cline.gen_active_calc()
    atoms.rattle(rattle, rng=np.random)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    init_velocities(atoms, tem)

    if dt is None:
        if (atoms.numbers == 1).any():
            dt = 0.25
        else:
            dt = 1.

    if ml_filter:
        md_atoms = FilterDeltas(atoms, shrink=ml_filter)
    else:
        md_atoms = atoms

    if dynamics.upper() == 'NPT':
        dyn = npt_dynamics(md_atoms, dt, tem, bulk_modulus, stress, mask,
                           trajectory, loginterval, append, tdamp, pdamp)
    elif dynamics.upper() == 'LANGEVIN':
        dyn = langevin_dynamics(md_atoms, dt, tem, friction, trajectory,
                                loginterval, append)

    if calc.meta is not None:
        dyn.attach(calc.meta.update)

    steps = int(picos*1000/dt) if picos > 0 else -picos
    dyn.run(steps)


def langevin_dynamics(atoms, dt, tem, friction, trajectory, loginterval, append):
    dyn = Langevin(atoms, dt*units.fs, temperature_K=tem, friction=friction, rng=np.random,
                   trajectory=trajectory, append_trajectory=append, loginterval=loginterval)
    return dyn


def npt_dynamics(atoms, dt, tem, bulk_modulus, stress, mask, trajectory, loginterval,
                 append, tdamp, pdamp):
    ttime = tdamp*units.fs
    ptime = pdamp*units.fs
    if bulk_modulus:
        pfactor = (ptime**2)*bulk_modulus*units.GPa
    else:
        pfactor = None
    configure_cell(atoms)
    dyn = NPT(atoms, dt*units.fs, temperature_K=tem, externalstress=stress*units.GPa,
              ttime=ttime, pfactor=pfactor, mask=mask, trajectory=trajectory,
              append_trajectory=append, loginterval=loginterval)
    return dyn


def configure_cell(atoms):
    if np.allclose(atoms.cell, 0.):
        atoms.center(vacuum=6.)
    make_cell_upper_triangular(atoms)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Machine Learning Molecular Dynamics (MLMD)')
    parser.add_argument('-i', '--input', default='POSCAR', type=str,
                        help='the initial coordinates of atoms, POSCAR, xyz, cif, etc.')
    parser.add_argument('-o', '--output', default='CONTCAR', type=str,
                        help='the final coordinates of atoms')
    args = parser.parse_args()
    if args.input.endswith('.traj'):
        atoms = read(args.input, -1)
    else:
        atoms = read(args.input)
    kwargs = cline.get_default_args(md)
    cline.update_args(kwargs)
    md(atoms, **kwargs)
    try:
        atoms.write(args.output)
    except:
        import warnings
        alt = 'md.final.xyz'
        msg = f'writing to {args.output} failed -> wrote {alt}'
        warnings.warn(msg)
        atoms.write(alt)
