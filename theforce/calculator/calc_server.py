import sys
from ase.io import read
from theforce.util.server import Server
import warnings
from theforce.util.util import date


def reserve_ofile(o, msg='reserved'):
    with open(o, 'w') as f:
        f.write(f'{date()} {msg}\n')


def get_calc(script, ref='calc'):
    scope = {}
    try:
        exec(open(script).read(), scope)
    except TypeError:
        exec(script.read(), scope)
    return scope[ref]


def calculate(_file, _calc):
    calc = _calc
    file = _file.decode("utf-8")
    if ':' in file:
        msg = file.split(':')
        if len(msg) == 2:
            i, o = msg
        elif len(msg) == 3:
            i, o, c = msg
            calc = get_calc(c)
        elif len(msg) == 4:
            i, o, c, ref = msg
            calc = get_calc(c, ref=ref)
        else:
            raise RuntimeError(f'message > 3 -> {msg}')
    else:
        i = o = file
    try:
        reserve_ofile(o)
        atoms = read(i)
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        atoms.get_forces()
        atoms.get_stress()
        atoms.write(o)
    except FileNotFoundError:
        warnings.warn(f'unable to read {i} -> calculation skipped')


if __name__ == '__main__':
    import argparse
    from theforce.util.ssh import clear_port

    parser = argparse.ArgumentParser(
        description='Starts a calculation server.')
    parser.add_argument('-ip', '--ip', default='localhost')
    parser.add_argument('-port', '--port', type=int, default=6666)
    parser.add_argument('-calc', '--calculator', default=None,
                        help=('If given, it should be a python script in which ' +
                              'a variable named calc is defined.'))
    args = parser.parse_args()

    if args.calculator is not None:
        calc = get_calc(args.calculator)
    else:
        calc = None

    clear_port(args.port)
    s = Server(args.ip, args.port, callback=calculate, args=(calc,))
    s.listen()
