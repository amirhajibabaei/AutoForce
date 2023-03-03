import warnings

import numpy as np


def cat(arrays, axis=0):
    lengths = [array.shape[axis] for array in arrays]
    cat = np.concatenate(arrays, axis=axis)
    spec = lengths + [axis]
    return cat, spec


def split(array, spec):
    return np.split(array, np.cumsum(spec[:-2]), spec[-1])


class SparseArray:
    def __init__(self, shape=(0,)):
        # configure the sparse axis (saxis)
        try:
            self.saxis = shape.index(0)
        except ValueError:
            raise RuntimeError(
                "No sparse axis is defined by setting it to 0 in the input shape!"
            )
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

        # make arrays
        _i, _j = np.broadcast_arrays(i, j)
        _v = np.asarray(v)

        # check if input is correct
        assert _i.ndim == 1 and _i.shape[0] == v.shape[self.saxis]
        assert all([a == b for a, b in zip(v.shape, self.shape) if b != 0])

        # check status and covert if needed
        if type(self.i) == np.ndarray:
            self._split()

        self.i += [_i]
        self.j += [_j]
        self.a += [_v]

    def _cat(self):
        if type(self.i) == list:
            self.i, self._ispec = cat(self.i)
            self.j, self._jspec = cat(self.j)
            self.a, self._aspec = cat(self.a, self.saxis)
        self.i_max = self.i.max()
        self.j_max = self.j.max()

    def _split(self):
        if type(self.i) == np.ndarray:
            self.i = split(self.i, self._ispec)
            self.j = split(self.j, self._jspec)
            self.a = split(self.a, self._aspec)
            del self._ispec, self._jspec, self._aspec

    def _sort(self, key=1):
        """key: 0->i, 1->j"""
        if type(self.i) == list:
            self._cat()

        # get argsort
        ij = [a.tolist() for a in [self.j, self.i]]
        if key == 0:
            ij = ij[1::-1]
        _, argsort = zip(
            *sorted(
                [([a, b], c) for a, b, c in zip(*[*ij, range(self.i.shape[0])])],
                key=lambda x: x[0],
            )
        )
        argsort = np.array(argsort)

        # sort tensors
        self.i = self.i[argsort]
        self.j = self.j[argsort]
        self.a = np.take(self.a, argsort, self.saxis)

        # get counts, and create split-sizes for future
        idx = [self.i, self.j][key]
        unique, count = np.unique(idx, return_counts=True)
        count = count.tolist()
        self._ispec = count + [0]
        self._jspec = count + [0]
        self._aspec = count + [self.saxis]


# testing -------------------------------------------------------------
def test_cat_split():
    # cat and split
    a = np.random.uniform(size=(10, 10, 3))
    b = np.random.uniform(size=(10, 8, 3))
    c = np.random.uniform(size=(10, 9, 3))
    t, spec = cat([a, b, c], 1)
    # print(t.shape)
    # print([a.shape for a in split(t, spec)])


def test_sparse():
    a = np.random.uniform(size=(3))
    b = np.random.uniform(size=(4))
    c = np.random.uniform(size=(5))
    S = SparseArray()
    S.add(1, list(range(3)), a)
    S.add(2, list(range(4)), b)
    S.add(3, list(range(5)), c)
    S._cat()
    # print(S.i.shape, S.j.shape, S.a.shape)

    a = np.random.uniform(size=(7, 3, 6))
    b = np.random.uniform(size=(7, 4, 6))
    c = np.random.uniform(size=(7, 5, 6))
    S = SparseArray(shape=(7, 0, 6))
    S.add(1, list(range(3)), a)
    S.add(2, list(range(4)), b)
    S.add(3, list(range(5)), c)
    S._cat()
    S.add(3, list(range(5)), c)
    S._cat()
    # print(S.i.shape, S.j.shape, S.a.shape)

    # test full sorting
    s = SparseArray(shape=(0,))
    s.add([4, 3, 2], 1, np.array([4, 3, 2]))
    s.add([3, 1], 2, np.array([3, 1]))
    s.add([2, 1, 4], 3, np.array([2, 1, 4]))
    s.add([1, 3], 4, np.array([1, 3]))
    s.add([2], 4, np.array([2]))

    s._sort(key=1)
    s._split()
    print(s.i)
    print(s.j)
    print(s.a)


if __name__ == "__main__":
    test_cat_split()
    test_sparse()
