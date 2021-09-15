# +
from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


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
