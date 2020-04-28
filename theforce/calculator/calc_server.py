import sys
from ase.io import read
from theforce.util.server import Server
import warnings
from theforce.util.util import date


def reserve_ofile(o, msg='reserved'):
    with open(o, 'w') as f:
        f.write(f'{date()} {msg}\n')


def get_calc(script):
    scope = {}
    try:
        exec(open(script).read(), scope)
    except TypeError:
        exec(script.read(), scope)
    return scope['calc']


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
    calc = sys.argv[1] if len(sys.argv) > 1 else 'calculator.py'
    if calc != 'None':
        calc = get_calc(calc)
    ip = sys.argv[2] if len(sys.argv) > 2 else 'localhost'
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 6666
    s = Server(ip, port, callback=calculate, args=(calc,))
    s.listen()
