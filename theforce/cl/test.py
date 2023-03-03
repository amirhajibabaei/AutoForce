# +
import warnings

from ase.io import Trajectory, read

import theforce.cl as cline


def test(*args, r="::", o="test.traj"):
    if "calculator" in cline.ARGS and cline.ARGS["calculator"] is not None:
        raise RuntimeError("set calculator = None in ARGS!")
    traj = Trajectory(o, "w")
    calc = cline.gen_active_calc()
    for arg in args:
        data = [read(arg)] if (r is None or ":" not in r) else read(arg, r)
        for atoms in data:
            atoms.set_calculator(calc)
            atoms.get_forces()
            if calc.rank == 0:
                traj.write(atoms)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the ML potential on input data")
    parser.add_argument("-i", "--input", nargs="*", type=str, help=".traj etc. files")
    parser.add_argument(
        "-r",
        "--read",
        type=str,
        default="::",
        help="index or [start]:[stop]:[step] e.g. 0 or -1 or ::10",
    )
    parser.add_argument("-o", "--output", type=str, default="test.traj", help=".traj")
    args = parser.parse_args()
    test(*args.input, r=args.read, o=args.output)
