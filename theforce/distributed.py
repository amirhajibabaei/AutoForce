# +
import torch

if torch.distributed.is_available():
    if torch.distributed.is_mpi_available():
        import torch.distributed as _dist
    else:
        import theforce._mpi4py as _dist
else:
    import theforce._mpi4py as _dist

group = _dist.group
is_initialized = _dist.is_initialized
init_process_group = _dist.init_process_group
broadcast = _dist.broadcast
get_world_size = _dist.get_world_size
get_rank = _dist.get_rank
barrier = _dist.barrier
all_reduce = _dist.all_reduce
ReduceOp = _dist.ReduceOp
