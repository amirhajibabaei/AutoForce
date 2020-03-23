import sys
from ase.io import read
from theforce.util.server import Server


def get_calc(script):
    scope = {}
    try:
        exec(open(script).read(), scope)
    except TypeError:
        exec(script.read(), scope)
    return scope['calc']


def calculate(_file, calc):
    file = _file.decode("utf-8")
    if '/' in file:
        i, o = file.split('/')
    else:
        i = o = file
    atoms = read(i)
    atoms.set_calculator(calc)
    atoms.get_forces()
    atoms.write(o)


if __name__ == '__main__':
    calc = get_calc(sys.argv[1] if len(sys.argv) > 1 else 'calculator.py')
    ip = sys.argv[2] if len(sys.argv) > 2 else 'localhost'
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 6666
    s = Server(ip, port, callback=calculate, args=(calc,))
    s.listen()
