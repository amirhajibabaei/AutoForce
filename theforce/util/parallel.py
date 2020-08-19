# +
from theforce.util.util import iterable
import torch.distributed as dist
import torch
import numpy as np
import functools
import warnings


def mpi_init(unify_randomness=True):
    """returns mpi WORLD"""
    dist.init_process_group('mpi')
    if unify_randomness:
        seed = torch.tensor(np.random.randint(2**32))
        dist.broadcast(seed, 0)
        np.random.seed(seed.numpy())
    return dist.group.WORLD


def index_gather(x, index, size=None):
    """currently only along dim 0 -> TODO: general dim"""
    if size is None:
        size = torch.tensor(max(index))
        dist.all_reduce(size, dist.ReduceOp.MAX)
        size = int(size) + 1
    _size = [s for s in x.size()]
    _size[0] = size
    _x = torch.zeros(*_size)
    _x[index] = x
    dist.all_reduce(_x)
    return _x


def balance_work(size, workers):
    # sizes
    a = size//workers
    b = size % workers
    work = [a+1 if j < b else a for j in range(workers)]
    # indices
    indices = []
    start = 0
    for chunk in work:
        indices += [(start, start+chunk)]
        start = start+chunk
    return indices


def method_forker(method):
    """
    works if:
    input shapes: [a], [b], [c], ...
    output shape: [a, b, c, ...]
    """

    @functools.wraps(method)
    def forker(self, *args, **kwargs):
        if hasattr(self, 'process_group'):
            # split work
            shape = [len(iterable(arg)) for arg in args]
            size = max(shape)
            dim = shape.index(size)
            workers = dist.get_world_size(group=self.process_group)
            if size < workers:  # TODO: warn or error?
                warnings.warn('size ({}) < workers ({})'.format(size, workers))
            indices = balance_work(size, workers)
            rank = dist.get_rank(group=self.process_group)
            start, end = indices[rank]
            t = method(self, *(arg[start:end] if d == dim else arg
                               for d, arg in enumerate(args)), **kwargs)

            # allocate and gather
            shapes = [torch.zeros(len(shape)).long() for _ in range(workers)]
            dist.all_gather(shapes, torch.tensor([s for s in t.size()]),
                            group=self.process_group)
            pieces = [torch.zeros(*sh).type(t.type()) for sh in shapes]
            pieces[rank] = t
            for k in range(workers):
                dist.broadcast(pieces[k], k, group=self.process_group)

            # concat
            out = torch.cat(pieces, dim=dim)
            return out
        else:
            return method(self, *args, **kwargs)

    return forker


def example():

    class Dummy:

        def __init__(self):
            pass

        @method_forker
        def f(self, x):
            return torch.cat(x)

        @method_forker
        def ff(self, x, xx):
            return torch.cat(x)[:, None] - torch.cat(xx)[None]

    model = Dummy()
    x = [torch.tensor(j).view(1) for j in range(5)]
    fo1 = model.f(x)
    ffo1 = model.ff(x, x)

    dist.init_process_group('mpi')
    model.process_group = dist.group.WORLD
    fo2 = model.f(x)
    ffo2 = model.ff(x, x)

    print((fo1 == fo2).all())
    print((ffo1 == ffo2).all())


if __name__ == '__main__':
    example()
