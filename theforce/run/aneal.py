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
from theforce.run.fly import suffixed_new
from theforce.util.ssh import forward_port, clear_port
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


def aneal(atoms=None, te=[100, 300, 1000], rc=7., lmax=3, nmax=3, eta=4, ediff=0.05, dt=2.,
          siz=10., maxat=256, stress=None, modulus=None, jumps=100, std=False, calc_args={},
          restrict=None, volatile=None, remote=None):
    """
    te -> Kelvin
    rc, siz -> Ang
    ediff -> eV
    dt -> fs
    stress, mosulus -> GPa (or None)
    """

    #
    init_process()

    if atoms is not None:
        #
        if std:
            atoms = standard_cell(atoms, symprec=0.1, angle_tolerance=10)

        # size
        if siz is not None:
            repeat = get_repeat_reciprocal(atoms, 1./siz)
            atoms = atoms.repeat(repeat)
        if atoms.get_number_of_atoms() > maxat:
            skip(f'too large: {len(atoms)} atoms')

    # kernel
    if atoms is None:
        kern = None
    else:
        numbers = np.unique(atoms.numbers)
        if len(numbers < 4):
            kern = default_kernel(numbers, cutoff=rc, lmax=lmax,
                                  nmax=nmax, exponent=eta)
        else:
            kern = UniversalSoapKernel(lmax, nmax, eta, rc)

    # calc
    calc = SocketCalculator(**calc_args)
    if remote is not None and dist.get_rank() == 0:
        clear_port(calc.port)
        forward_port(calc.port, remote, ip=calc.ip)

    # params
    kw = dict(dt=dt, calc=calc, ediff=ediff,
              group=dist.group.WORLD, kern=kern)
    if stress is not None and modulus is not None:
        kw['ext_stress'] = stress*GPa
        kw['pfactor'] = (20, modulus*GPa)

    # run
    _, suff = suffixed_new('model')
    k = letters.index(suff)
    for s, t in zip(*[letters[k:k+len(te)], te]):
        if not os.path.isdir(f'model_{s}'):
            try:
                if s == suff:
                    fly(t, jumps, atoms=atoms, lf_kwargs=dict(
                        restrict=restrict, volatile=volatile), **kw)
                else:
                    fly(t, jumps, lf_kwargs=dict(restrict=restrict), **kw)
            except:
                goodbye()
                raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Machine Learning of the PES by anealing')

    # arguments
    parser.add_argument('-a', '--atoms', default=None,
                        help='path to a file to read atoms')
    parser.add_argument('-r', '--repeat', default='1,1,1',
                        help='repeat to make a supercell')
    parser.add_argument('-t', '--temperatures', default='300,600,1000',
                        help="e.g. 300,600,1000")
    parser.add_argument('-p', '--pressure', type=float, default=None,
                        help="GPa")
    parser.add_argument('-m', '--modulus', type=float, default=None,
                        help="GPa")
    parser.add_argument('-s', '--size', default='10.',
                        help="Angstrom, also could be None")
    parser.add_argument('-j', '--jumps', type=int, default=100,
                        help="training trials")
    parser.add_argument('-std', '--standard', type=int, choices=(0, 1), default=0,
                        help="standard cell transform 0: no (default) 1: yes")
    parser.add_argument('-res', '--restrict', default=None,
                        help="atomic numbers e.g. 3,15,16")
    parser.add_argument('-v', '--volatile', type=int, default=None,
                        help="treat as volatile for v extremums")

    # socket calculator
    parser.add_argument('-ip', '--ip', default='localhost')
    parser.add_argument('-port', '--port', type=int, default=6666)
    parser.add_argument('-calc', '--calculator', default=None)
    parser.add_argument('-rem', '--remote', default=None)

    # parse
    args = parser.parse_args()

    #
    if args.atoms is None:
        atoms = None
    else:
        repeat = [int(i) for i in args.repeat.split(',')]
        atoms = (read(args.atoms, -1) if args.atoms.endswith('.traj')
                 else read(args.atoms)).repeat(repeat)
    #
    tempretures = [float(t) for t in args.temperatures.split(',')]
    #
    calc_args = dict(ip=args.ip, port=args.port, script=args.calc)
    #
    if ((args.pressure is None and args.modulus is not None) or
            (args.pressure is not None and args.modulus is None)):
        raise RuntimeError('both pressure and modulus should be given!')

    #
    restrict = (None if args.restrict is None else
                [int(z) for z in args.restrict.split(',')])

    # run
    aneal(atoms=atoms, te=tempretures, stress=args.pressure, modulus=args.modulus,
          siz=eval(args.size), jumps=args.jumps, calc_args=calc_args, std=args.standard,
          restrict=restrict, volatile=args.volatile, remote=args.remote)
