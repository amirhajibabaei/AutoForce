
# coding: utf-8

# In[ ]:


import numpy as np
import warnings


def cat(arrays, axis=0):
    lengths = [array.shape[axis] for array in arrays]
    cat = np.concatenate(arrays, axis=axis)
    spec = lengths + [axis]
    return cat, spec


def split(array, spec):
    return np.split(array, np.cumsum(spec[:-2]), spec[-1])


class SparseArray:

    def __init__(self, shape=(0,)):
        # configure the sparse axis
        try:
            self.axis = shape.index(0)
        except ValueError:
            raise RuntimeError(
                "No sparse axis is defined by setting it to 0 in the input shape!")
        if shape.count(0) > 1:
            warnings.warn("Multiple 0's in the input shape")
        self.shape = shape

        # data holders
        self.i, self.j, self.a = [], [], []

    def add(self, i, j, v):
        _i, _j = np.broadcast_arrays(i, j)

        # check if input is correct
        assert _i.ndim == 1 and _i.shape[0] == v.shape[self.axis]
        assert all([a == b for a, b in zip(v.shape, self.shape) if b != 0])

        # check status and covert if needed
        if type(self.i) == np.ndarray:
            self._split()

        self.i += [_i]
        self.j += [_j]
        self.a += [v]

    def _cat(self):
        if type(self.i) == list:
            self.i, self._ispec = cat(self.i)
            self.j, self._jspec = cat(self.j)
            self.a, self._aspec = cat(self.a, self.axis)

    def _split(self):
        if type(self.i) == np.ndarray:
            self.i = split(self.i, self._ispec)
            self.j = split(self.j, self._jspec)
            self.a = split(self.a, self._aspec)
            del self._ispec, self._jspec, self._aspec


# testing -------------------------------------------------------------
def test_cat_split():
    # cat and split
    a = np.random.uniform(size=(10, 10, 3))
    b = np.random.uniform(size=(10, 8, 3))
    c = np.random.uniform(size=(10, 9, 3))
    t, spec = cat([a, b, c], 1)
    print(t.shape)
    print([a.shape for a in split(t, spec)])


def test_sparse():
    a = np.random.uniform(size=(3))
    b = np.random.uniform(size=(4))
    c = np.random.uniform(size=(5))
    S = SparseArray()
    S.add(1, list(range(3)), a)
    S.add(2, list(range(4)), b)
    S.add(3, list(range(5)), c)
    S._cat()
    print(S.i.shape, S.j.shape, S.a.shape)

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
    print(S.i.shape, S.j.shape, S.a.shape)


if __name__ == '__main__':
    test_cat_split()
    test_sparse()

