import torch
import functools
from theforce.util.util import iterable
import warnings


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
            workers = torch.distributed.get_world_size(
                group=self.process_group)
            if size < workers:  # TODO: warn or error?
                warnings.warn('size ({}) < workers ({})'.format(size, workers))
            indices = balance_work(size, workers)
            rank = torch.distributed.get_rank(group=self.process_group)
            start, end = indices[rank]
            t = method(self, *(arg[start:end] if d == dim else arg
                               for d, arg in enumerate(args)), **kwargs)

            # allocate and gather
            shapes = [torch.zeros(len(shape)).long() for _ in range(workers)]
            torch.distributed.all_gather(shapes, torch.tensor([s for s in t.size()]),
                                         group=self.process_group)
            pieces = [torch.zeros(*sh).type(t.type()) for sh in shapes]
            pieces[rank] = t
            for k in range(workers):
                torch.distributed.broadcast(
                    pieces[k], k, group=self.process_group)

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

    torch.distributed.init_process_group('mpi')
    model.process_group = torch.distributed.group.WORLD
    fo2 = model.f(x)
    ffo2 = model.ff(x, x)

    print((fo1 == fo2).all())
    print((ffo1 == ffo2).all())


if __name__ == '__main__':
    example()

