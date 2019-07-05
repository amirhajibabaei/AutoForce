
# coding: utf-8

# In[ ]:


import torch
import warnings


def nan_to_num(t, num=0.):
    return torch.where(torch.isnan(t), torch.as_tensor(num), t)


def zero_to_tiny(t, tiny=1e-6):
    return torch.where(t == 0., torch.as_tensor(tiny), t)


def cat(tensors, dim=0):
    lengths = [tensor.size(dim) for tensor in tensors]
    cat = torch.cat(tensors, dim=dim)
    spec = torch.LongTensor(lengths + [dim])
    return cat, spec


def split(tensor, spec):
    return torch.split(tensor, spec[:-1].tolist(), spec[-1])


def stretch_tensor(a, dims):
    size = list(a.size())
    added = 0
    for dim in sorted(dims):
        size.insert(added+dim, 1)
        added += 1
    return a.view(*size)


class SparseTensor:

    def __init__(self, shape=(0,)):
        # configure the sparse dim (sdim)
        try:
            self.sdim = shape.index(0)
        except ValueError:
            raise RuntimeError(
                "No sparse dim is defined by setting it to 0 in the input shape!")
        if shape.count(0) > 1:
            warnings.warn("Multiple 0's in the input shape")
        self.shape = shape

        # data holders
        self.i, self.j, self.a = [], [], []

    def add(self, i, j, v):
        # many inputs at once
        if type(i) == type(j) == type(v) == list:
            for a, b, c in zip(*[i, j, v]):
                self.add(a, b, c)
            return

        # tensorify
        _i, _j = torch.broadcast_tensors(torch.as_tensor(i),
                                         torch.as_tensor(j))
        _v = torch.as_tensor(v)

        # check if input is correct
        assert _i.dim() == 1 and _i.size(0) == _v.size(self.sdim)
        assert all([a == b for a, b in zip(v.shape, self.shape) if b != 0])

        # check status and covert if needed
        if type(self.i) == torch.Tensor:
            self._split()

        self.i += [_i]
        self.j += [_j]
        self.a += [_v]

    def _cat(self):
        if type(self.i) == list:
            self.i, self._ispec = cat(self.i)
            self.j, self._jspec = cat(self.j)
            self.a, self._aspec = cat(self.a, self.sdim)
        self.i_max = self.i.max()
        self.j_max = self.j.max()

    def _split(self):
        if type(self.i) == torch.Tensor:
            self.i = list(split(self.i, self._ispec))
            self.j = list(split(self.j, self._jspec))
            self.a = list(split(self.a, self._aspec))
            del self._ispec, self._jspec, self._aspec

    def _sort(self, key=1):
        """ key: 0->i, 1->j """
        if type(self.i) == list:
            self._cat()

        # get argsort
        ij = [a.tolist() for a in [self.j, self.i]]
        if key == 0:
            ij = ij[1::-1]
        _, argsort = zip(*sorted([([a, b], c) for a, b, c in
                                  zip(*[*ij, range(self.i.size(0))])],
                                 key=lambda x: x[0]))
        argsort = torch.LongTensor(argsort)

        # sort tensors
        self.i = self.i[argsort]
        self.j = self.j[argsort]
        self.a = torch.index_select(self.a, self.sdim, argsort)

        # get counts, and create split-sizes for future
        idx = [self.i, self.j][key]
        unique = idx.unique(sorted=True)
        count = torch.stack([(idx == e).sum() for e in unique]).tolist()
        self._ispec = torch.LongTensor(count + [0])
        self._jspec = torch.LongTensor(count + [0])
        self._aspec = torch.LongTensor(count + [self.sdim])


# -------------------------------------------------------------
def test():
    # cat and split
    a = torch.rand(10, 7, 3)
    b = torch.rand(10, 8, 3)
    c = torch.rand(10, 9, 3)
    t, spec = cat([a, b, c], 1)
    #print([a.shape for a in split(t, spec)])


def test_sparse():
    a = torch.rand(3)
    b = torch.rand(4)
    c = torch.rand(5)
    S = SparseTensor()
    S.add(1, list(range(3)), a)
    S.add(2, list(range(4)), b)
    S.add(3, list(range(5)), c)
    S.add([1, 1, 1], [list(range(3)), list(range(3)),
                      list(range(3))], [a, a, a])
    S._cat()
    #print(S.i.shape, S.j.shape, S.a.shape)

    a = torch.rand(7, 3, 6)
    b = torch.rand(7, 4, 6)
    c = torch.rand(7, 5, 6)
    S = SparseTensor(shape=(7, 0, 6))
    S.add(1, list(range(3)), a)
    S.add(2, list(range(4)), b)
    S.add(3, list(range(5)), c)
    S._cat()
    S.add(3, list(range(5)), c)
    S._cat()
    #print(S.i.shape, S.j.shape, S.a.shape)

    # test full sorting
    s = SparseTensor(shape=(0,))
    s.add([4, 3, 2], 1, torch.tensor([4, 3, 2]))
    s.add([3, 1], 2, torch.tensor([3, 1]))
    s.add([2, 1, 4], 3, torch.tensor([2, 1, 4]))
    s.add([1, 3], 4, torch.tensor([1, 3]))
    s.add([2], 4, torch.tensor([2]))

    s._sort(key=1)
    s._split()
    print(s.i)
    print(s.j)
    print(s.a)


if __name__ == '__main__':
    test()
    test_sparse()

