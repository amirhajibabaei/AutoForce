# +
import warnings

from ase.io import Trajectory, read

import theforce.cl as cline


def offline(*args, r=None, o="offline.traj"):
    if "calculator" not in cline.ARGS or cline.ARGS["calculator"] is None:
        raise RuntimeError("set a calculator in ARGS!")
    traj = Trajectory(o, "w")
    calc = cline.gen_active_calc()
    for arg in args:
        data = [read(arg)] if (r is None or ":" not in r) else read(arg, r)
        for atoms in data:
            atoms.calc = calc
            atoms.get_forces()
            if calc.rank == 0:
                traj.write(atoms)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the ML potential on input configurations. "
        "Ab initio calculations will be performed as needed. "
    )
    parser.add_argument("-i", "--input", nargs="*", type=str, help=".traj etc. files")
    parser.add_argument(
        "-r",
        "--read",
        type=str,
        default="::",
        help="index or [start]:[stop]:[step] e.g. 0 or -1 or ::10",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="offline.traj", help=".traj"
    )
    args = parser.parse_args()
    offline(*args.input, r=args.read, o=args.output)
