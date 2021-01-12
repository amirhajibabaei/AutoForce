# +
from theforce.util.util import get_default_args
from theforce.util.parallel import mpi_init
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator
import os


def strip(line):
    if '#' in line:
        return line[:line.index('#')].strip()
    else:
        return line.strip()


def _calc(name):
    if name.upper() == 'VASP':
        from theforce.calculator import vasp
        calc_script = vasp.__file__
    return calc_script


def gen_active_calc(**over):
    kwargs = get_default_args(ActiveCalculator.__init__)
    for kw in kwargs:
        if kw in ARGS:
            kwargs[kw] = ARGS[kw]
        if kw in over:
            kwargs[kw] = over[kw]
    return ActiveCalculator(**kwargs)


ARGS = {'process_group': mpi_init()}
if os.path.isfile('ARGS'):
    lines = [strip(line) for line in
             open('ARGS').readlines()]
    lines = ','.join(filter(''.__ne__, lines))
    ARGS = eval(f'dict({lines})')
    if ARGS['calculator'] is not None:
        calc_script = _calc(ARGS['calculator'])
        ARGS['calculator'] = SocketCalculator(script=calc_script)
