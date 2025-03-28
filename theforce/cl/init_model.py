# +
import os

import numpy as np
from ase.io import Trajectory, read

import theforce.cl as cline


def init_model(atoms, samples=5, rattle=0.05, trajectory="init.traj"):
    """
    atoms:        ASE atoms
    samples:      number of samples
    rattle:       stdev for random displacements
    trajectory:   traj file name
    """

    calc = cline.gen_active_calc()
    master = calc.rank == 0
    if master:
        traj = Trajectory(trajectory, "w")
    for _ in range(samples):
        tmp = atoms.copy()
        tmp.rattle(rattle, rng=np.random)
        tmp.calc = calc
        tmp.get_potential_energy()
        if master:
            traj.write(tmp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Initializes a Machine Learning potential by random displacements"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="POSCAR",
        type=str,
        help="the initial coordinates of atoms, POSCAR, xyz, cif, etc.",
    )
    args = parser.parse_args()
    atoms = read(args.input)
    kwargs = cline.get_default_args(init_model)
    cline.update_args(kwargs)
    init_model(atoms, **kwargs)
