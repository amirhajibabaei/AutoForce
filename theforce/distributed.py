# +
import torch.distributed as _dist


group = _dist.group
is_initialized = _dist.is_initialized
init_process_group = _dist.init_process_group
broadcast = _dist.broadcast
get_world_size = _dist.get_world_size
get_rank = _dist.get_rank
barrier = _dist.barrier
all_reduce = _dist.all_reduce
