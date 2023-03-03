# +
import warnings

from ase.io import read

import theforce.cl as cline


def train(*args, r=None):
    calc = cline.gen_active_calc()
    for arg in args:
        if arg.endswith(".sgpr"):
            if r is not None and r != "::":
                try:
                    ndata = int(r)
                except:
                    raise RuntimeError(
                        "For .sgpr files use -r with an integer (e.g. -r 100)"
                    )
            else:
                ndata = None
            calc.include_tape(arg, ndata=ndata)
        else:
            data = read(arg) if r is None else read(arg, r)
            calc.include_data(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ML potential using data")
    parser.add_argument("-i", "--input", nargs="*", type=str, help=".traj or .sgpr")
    parser.add_argument(
        "-r",
        "--read",
        type=str,
        default="::",
        help="index or [start]:[stop]:[step] e.g. 0 or -1 or ::10",
    )
    args = parser.parse_args()
    train(*args.input, r=args.read)
