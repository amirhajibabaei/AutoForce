# +
import theforce.cl as cline
from ase.io import read
import warnings


def train(*args, r=None):
    calc = cline.gen_active_calc()
    for arg in args:
        if arg.endswith('.sgpr'):
            if r is not None:
                w = f'-r {r} option is ignored for {arg}'
                warnings.warn(w)
            calc.include_tape(arg)
        else:
            data = read(arg) if r is None else read(arg, r)
            calc.include_data(data)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train ML potential using data')
    parser.add_argument('-i', '--input', nargs='*', type=str,
                        help='.traj or .sgpr')
    parser.add_argument('-r', '--read', type=str, default='::',
                        help='index or [start]:[stop]:[step] e.g. 0 or -1 or ::10')
    args = parser.parse_args()
    train(*args.input, r=args.read)
