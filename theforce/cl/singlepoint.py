# +
import theforce.cl as cline
from ase.io import read


def single_point(i, o):
    atoms = read(i)
    calc = cline.gen_active_calc()
    atoms.set_calculator(calc)
    atoms.get_forces()
    if calc.rank == 0:
        atoms.write(o)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='single-point ML energy & forces')
    parser.add_argument('-i', '--input', type=str, default='POSCAR',
                        help='input file: name.xyz, POSCAR, ...')
    parser.add_argument('-o', '--output', type=str, default='singlepoint.xyz',
                        help='output file: out.xyz, CONTCAR, ...')
    args = parser.parse_args()
    single_point(args.input, args.output)
