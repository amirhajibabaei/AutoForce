# +
from collections import Counter
import numpy as np
from theforce.io.sgprio import SgprIO


def no_sgpr_duplicates(tape):
    """
    tape: Path to a .sgpr file.

    The file will be modified in-place, with duplicate data eliminated.
    """

    # read
    data = SgprIO(tape).read()

    # filter
    n = len(data)
    unique = []
    for i in range(0, n):
        is_unique = True
        for j in range(0, i):
            if the_same(data[i], data[j]):
                is_unique = False
                break
        if is_unique:
            unique += [data[i]]

    # write
    with open(tape, 'w') as f:
        f.write('\n')
    new = SgprIO(tape)
    for a, b in unique:
        new.write(b)

    #
    print(f'unique count: {count(unique)}')
    return unique


def same_atoms(a, b):
    check = (len(a) == len(b) and
             (a.numbers == b.numbers).all() and
             np.allclose(a.positions, b.positions) and
             np.allclose(a.cell, b.cell) and
             (a.pbc == b.pbc).all()
             )
    return check


def same_locals(a, b):
    check = (a.number == b.number and
             len(a._b) == len(b._b) and
             (a._b == b._b).all() and
             a._r.allclose(b._r)
             )
    return check


def the_same(a, b, eq=None):
    if eq is None:
        eq = {'atoms': same_atoms, 'local': same_locals}
    check = (a[0] == b[0] and
             eq[a[0]](a[1], b[1])
             )
    return check


def count(data):
    c = Counter()
    for a, b in data:
        c[a] += 1
    return c


if __name__ == '__main__':
    import sys

    tapes = sys.argv[1:]
    for tape in tapes:
        no_sgpr_duplicates(tape)
