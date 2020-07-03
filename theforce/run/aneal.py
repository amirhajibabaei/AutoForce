from theforce.run.fly import fly, default_kernel, numpy_same_random_seed
from theforce.similarity.universal import UniversalSoapKernel
from theforce.calculator.posterior import SocketCalculator
from theforce.util.aseutil import get_repeat_reciprocal
from theforce.util.pmgen import standard_cell
from theforce.util.util import date
from ase.units import Pascal, GPa
import torch.distributed as dist
from ase.io import read
import numpy as np
from string import ascii_lowercase as letters
import sys
import os
import atexit

# training-in-progress: tip.lock
lock = 'tip.lock'


def init_process():
    dist.init_process_group('mpi')
    if os.path.isfile(lock):
        print(f'{lock} -> exit')
        exit = True
    else:
        exit = False
    dist.barrier()
    if exit:
        sys.exit()
    if dist.get_rank() == 0:
        with open(lock, 'w') as f:
            f.write(f'running since {date()}\n')
    dist.barrier()
    numpy_same_random_seed()


@atexit.register
def goodbye():
    os.system(f'rm -f {lock}')


def skip(msg):
    if dist.get_rank() == 0:
        with open('skipped', 'w') as f:
            f.write(f'{msg}\n')
        goodbye()
    dist.barrier()
    sys.exit()


def aneal(atoms, te=[100, 300, 1000], rc=7., lmax=3, nmax=3, eta=4, ediff=0.05, dt=2.,
          siz=10., maxat=256, stress=None, modulus=None, jumps=100, std=True, calc_args={}):
    """
    te -> Kelvin
    rc, siz -> Ang
    ediff -> eV
    dt -> fs
    stress, mosulus -> GPa (or None)
    """

    #
    init_process()

    #
    if std:
        atoms = standard_cell(atoms, symprec=0.1, angle_tolerance=10)
    numbers = np.unique(atoms.numbers)

    # kernel
    if len(numbers < 4):
        kern = default_kernel(numbers, cutoff=rc, lmax=lmax,
                              nmax=nmax, exponent=eta)
    else:
        kern = UniversalSoapKernel(lmax, nmax, eta, rc)

    # params
    kw = dict(dt=dt, calc=SocketCalculator(**calc_args), ediff=ediff,
              group=dist.group.WORLD, kern=kern)
    if stress is not None and modulus is not None:
        kw['ext_stress'] = stress*GPa
        kw['pfactor'] = (20, modulus*GPa)

    # size
    repeat = get_repeat_reciprocal(atoms, 1./siz)
    atoms = atoms.repeat(repeat)
    if atoms.get_number_of_atoms() > maxat:
        skip(f'too large: {len(atoms)} atoms')

    # run
    for s, t in zip(*[letters[:len(te)], te]):
        if not os.path.isdir(f'model_{s}'):
            try:
                if s == 'a':
                    fly(t, jumps, atoms=atoms, **kw)
                else:
                    fly(t, jumps, **kw)
            except:
                goodbye()
                raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Machine Learning of the PES by anealing')
    parser.add_argument('atoms')
    # temperatures and pressures
    parser.add_argument('-t', '--temperatures', default='100,300,1000')
    parser.add_argument('-p', '--pressure', type=float, default=None)
    parser.add_argument('-m', '--modulus', type=float, default=None)
    parser.add_argument('-j', '--jumps', type=int, default=100)
    # socket calculator
    parser.add_argument('-ip', '--ip', default='localhost')
    parser.add_argument('-port', '--port', type=int, default=6666)
    parser.add_argument('-calc', '--calc', default=None)

    args = parser.parse_args()
    atoms = (read(args.atoms, -1) if arg.atoms.endswith('.traj')
             else read(args.atoms))
    tempretures = [float(t) for t in args.te.split(',')]
    calc_args = dict(ip=args.ip, port=args.port, script=args.calc)

    aneal(atoms, te=tempretures, stress=args.stress, modulus=args.mod,
          jumps=args.jumps, calc_args=calc_args)
