# +
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.util.parallel import mpi_init
from theforce.util.aseutil import init_velocities, make_cell_upper_triangular
from ase.md.npt import NPT
from ase.io import read
from ase import units
import numpy as np
import os


def mlmd(atoms, covariance=None, calc_script=None, dt=None, tem=300., picos=100,
         bulk_modulus=None, stress=0., mask=None, group=None, tape='model.sgpr',
         trajectory='md.traj', loginterval=1, append=False, rattle=0.05, 
         ediff=0.041, fdiff=0.1, coveps=1e-4):
    """
    atoms:        ASE atoms
    covariance:   kernel or model
    calc_script:  path to a python script where the DFT calculator is defined
    dt:           time-step in fs
    tem:          temperature in Kelvin
    picos:        pico-seconds for md
    bulk_modulus: bulk_modulus for NPT simulations. if None, NVT is performed
    stress:       external stress (GPa) for NPT
    mask:         see ase.npt.NPT
    group:        process-group for MPI. if None, it is initiated
    tape:         checkpoint for the ML potential
    trajectory:   traj file name
    loginterval:  for traj file
    append:       append to traj file
    rattle:       rattle atoms at initial step (recommended ~0.05)
    ediff:        energy sensitivity
    fdiff:        force sensitivity
    coveps:       skips ML update if covloss < coveps
    """
    # set calculator
    if calc_script is not None:
        calc_script = SocketCalculator(script=calc_script)
    calc = ActiveCalculator(covariance=covariance,
                            calculator=calc_script,
                            process_group=group or mpi_init(),
                            tape=tape, fdiff=fdiff, ediff=ediff, coveps=coveps
                            )

    # process atoms
    if type(atoms) == str:
        if atoms.startswith('export'):
            folder = atoms.split('-')[1] if '-' in atoms else 'model'
            calc.model.to_folder(folder)
            return
        else:
            atoms = read(atoms)
    atoms.rattle(rattle, rng=np.random)  # rattle after mpi_init
    atoms.set_calculator(calc)

    # define and run Nose-Hoover dynamics
    if dt is None:
        if (atoms.numbers == 1).any():
            dt = 0.25
        else:
            dt = 1.
    ttime = 25*units.fs
    ptime = 100*units.fs
    if bulk_modulus:
        pfactor = (ptime**2)*bulk_modulus*units.GPa
    else:
        pfactor = None
    init_velocities(atoms, tem)
    make_cell_upper_triangular(atoms)
    filtered = FilterDeltas(atoms)
    steps_for_1ps = int(1000/dt)
    dyn = NPT(filtered, dt*units.fs, tem*units.kB, stress*units.GPa,
              ttime, pfactor, mask=mask, trajectory=trajectory,
              append_trajectory=append, loginterval=loginterval)
    dyn.run(picos*steps_for_1ps)


def read_md(file='MD'):
    if os.path.isfile(file):
        lines = ','.join([line.strip() for line in open(file).readlines()])
        return eval(f'dict({lines})')
    else:
        return {}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Machine Learning Molecular Dynamics (MLMD)')
    parser.add_argument('-i', '--input', default='POSCAR', type=str,
                        help='a command or the initial coordinates of atoms')
    args = parser.parse_args()
    mlmd(args.input, **read_md('MD'))
