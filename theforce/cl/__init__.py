# +
from theforce.util.util import get_default_args
from theforce.util.parallel import mpi_init
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator, kcal_mol, inf, SeSoapKernel, DefaultRadii, ActiveMeta
from theforce.calculator.meta import Meta, Posvar, Qlvar, Catvar
import torch
import theforce.distributed as dist
import os
import time
import atexit


def strip(line):
    if '#' in line:
        return line[:line.index('#')].strip()
    else:
        return line.strip()


def _calc(name):
    caps = name.upper()
    if caps == 'VASP':
        from theforce.calculator import vasp
        calc_script = vasp.__file__
    elif caps == 'GAUSSIAN':
        from theforce.calculator import gaussian
        calc_script = gaussian.__file__
    elif caps == 'EMT':
        from theforce.calculator import emt
        calc_script = emt.__file__
    else:
        raise RuntimeError(f'calculator {caps} is not implemented')
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


def print_stop_time():
    dist.barrier()
    rank = dist.get_rank()
    world = dist.get_world_size()
    threads = torch.get_num_threads()
    stop_time = time.time() - start_time
    for i in range(world):
        if rank == i:
            print(
                f'\tstopwatch (process: {rank}/{world}, threads: {threads}): \t {stop_time} seconds')
        dist.barrier()


# at exit
start_time = time.time()
atexit.register(print_stop_time)

# ARGS
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
