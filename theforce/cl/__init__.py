# +
from theforce.util.util import get_default_args
from theforce.util.parallel import mpi_init
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator, kcal_mol, inf, SeSoapKernel, DefaultRadii
from theforce.calculator.meta import Meta, Posvar, Qlvar, Catvar
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


def update_args(kwargs, source=None):
    if source is None:
        source = ARGS
    for kw in kwargs:
        if kw in source:
            kwargs[kw] = source[kw]


def gen_active_calc(**over):
    kwargs = get_default_args(ActiveCalculator.__init__)
    update_args(kwargs)
    update_args(kwargs, source=over)
    return ActiveCalculator(**kwargs)


ARGS = {}
if os.path.isfile('ARGS'):
    lines = [strip(line) for line in
             open('ARGS').readlines()]
    lines = ','.join(filter(''.__ne__, lines))
    ARGS.update(eval(f'dict({lines})'))
    if 'calculator' in ARGS and ARGS['calculator'] is not None:
        calc_script = _calc(ARGS['calculator'])
        ARGS['calculator'] = SocketCalculator(script=calc_script)
seed = ARGS['seed'] if 'seed' in ARGS else None
ARGS['process_group'] = mpi_init(seed=seed)
