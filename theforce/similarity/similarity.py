# +
from torch.nn import Module
from torch import cat
from theforce.util.util import iterable
from theforce.util.caching import method_caching
from theforce.util.parallel import method_forker
import torch


class SimilarityKernel(Module):

    def __init__(self, kernel):
        super().__init__()
        self.kern = kernel
        self.params = self.kern.params

    @method_forker
    def forward(self, first, second, operation='func'):
        mat = [torch.empty(0)]
        for a in iterable(first):
            raw = [torch.empty(0)]
            for b in iterable(second):
                raw.append(getattr(self, operation)(a, b))
            mat.append(torch.cat(raw, dim=1))
        mat = torch.cat(mat, dim=0)
        if mat.numel() == 0:
            shape = (0, len(iterable(second)))
            # this shape is not general, but is is
            # chosen consistent with other parts
            mat = mat.view(*shape)
        return mat

    @method_forker
    def diag(self, first, operation='func'):
        return cat([getattr(self, operation+'diag')(a) for a in iterable(first)])

    @method_forker
    def funcdiag(self, first):
        return self.func(first, first).view(1)

    @method_caching
    def func(self, p, q):
        return self.get_func(p, q) + lone_atoms(p, q)

    @method_caching
    def leftgrad(self, p, q):
        return self.get_leftgrad(p, q)

    @method_caching
    def rightgrad(self, p, q):
        return self.get_rightgrad(p, q)

    @method_caching
    def gradgrad(self, p, q):
        return self.get_gradgrad(p, q)

    @method_caching
    def gradgraddiag(self, p):
        return self.get_gradgraddiag(p)

    def save_for_later(self, loc, keyvals):
        for key, val in keyvals.items():
            setattr(loc, self.name+'_'+key, val)

    def saved(self, atoms_or_loc, key):
        attr = self.name+'_'+key
        try:
            return torch.cat([loc.__dict__[attr] for loc in atoms_or_loc] + [torch.empty(0)])
        except TypeError:
            return atoms_or_loc.__dict__[attr]

    @property
    def state_args(self):
        return self.kern.state

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

    def __repr__(self):
        return self.state


def lone_atoms(_p, _q):
    k = 0
    for p in iterable(_p):
        if p._b.size(0) > 0:
            continue
        for q in iterable(_q):
            if p.number != q.number or q._b.size(0) > 0:
                continue
            k += 1
    return torch.tensor([[float(k)]])
