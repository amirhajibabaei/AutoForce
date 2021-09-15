# +
from mpi4py import MPI
import numpy as np
import sys


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
# if mpi4py is available in sys.modules,
# ase parallelization is activated which
# apparently has conflicts with theforce
# since no ase parallelization is assumed.
# The right way to corrcet the code is
# to search for all ase function invocations
# and consider parallelism case by case.
# For now, the ugly solution is:
# sys.modules['_mpi4py'] = sys.modules['mpi4py']
del sys.modules['mpi4py']


def is_initialized():
    return True


class group:
    WORLD = comm


class ReduceOp:
    MAX = MPI.MAX
    SUM = MPI.SUM


def init_process_group(arg='mpi'):
    assert arg == 'mpi'


def get_world_size(group=comm):
    return size


def get_rank(group=comm):
    return rank


def broadcast(data, src):
    a = data.detach().numpy()
    comm.Bcast(a, src)


def all_reduce(data, op=ReduceOp.SUM):
    a = data.detach().numpy().reshape(-1)
    b = np.zeros_like(a)
    comm.Allreduce(a, b, op)
    a[:] = b[:]


def barrier():
    comm.Barrier()
