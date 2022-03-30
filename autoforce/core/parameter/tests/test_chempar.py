# +
import autoforce.cfg as cfg
from autoforce.core import ChemPar, ReducedPar
from autoforce.functions import FiniteRange
from torch import Tensor
import torch


def test_ChemPar() -> bool:

    def tensor(*args, **kwargs):
        return torch.tensor(*args, **kwargs, dtype=cfg.float_t)

    #
    s = ChemPar(values={1: 1.})
    assert s[(1,)] == s[1] == 1. and s[2] == None
    s[:] = 3.
    assert s[1] == s[2] == 3.

    #
    s = ChemPar(values={1: 1.}, default=3.)
    assert s[1] == 1. and s[54345] == 3.
    a = torch.tensor([1, 2])
    assert s(a).allclose(tensor([1., 3.]))

    #
    s = ChemPar(values={(1, 2): 1.})
    assert s[1, 2] == s[2, 1] == s[(1, 2)] == s[(
        2, 1)] == 1. and s[1, 3] == None
    s[:] = 2.
    assert s[117, 75] == 2.

    #
    s = ChemPar(values={(1, 2): 1.}, permsym=False)
    assert s[1, 2] == 1. and s[2, 1] == None

    #
    s = ChemPar(values={(1, 2): 1.}, default=3., permsym=False)
    assert s[1, 2] == 1. and s[2, 1] == 3.

    #
    s = ChemPar(values={(1, 2): 1., (2, 1): 2.}, default=3., permsym=False)
    assert s[1, 2] == 1. and s[2, 1] == 2.

    #
    s = ChemPar(keylen=2)
    s[:] = 1.
    s[1, 2] = 2.
    assert s[1, 1] == s[2, 2] == 1. and s[1, 2] == s[2, 1] == 2.

    #
    s = ChemPar(values={(1, 2): 1., (2, 2): 7.}, default=3.)
    a = torch.tensor(1)
    b = torch.tensor([1, 2, 3])
    assert (s(a, b) == tensor([3., 1., 3.])).all()
    assert (s(b, b) == tensor([3., 7., 3.])).all()

    s.as_dict([1])

    #
    const = FiniteRange(0., 10.)
    s = ChemPar(values={(1, 2): 1., (2, 2): 7.}, default=3., bijection=const)
    assert s(a, b).allclose(tensor([3., 1., 3.]))
    assert s(b, b).allclose(tensor([3., 7., 3.]))

    return True


def test_ReducedPar():

    a = ChemPar(values={(1, 2): 1., (2, 2): 7.}, default=3.)
    b = ChemPar(values={(1, 2): 8., (2, 2): 2.}, default=4.)
    c = ReducedPar(a, b, op=max)
    d = c.as_dict([1, 2, 3], float)
    assert d[(1, 1)] == 4.
    assert d[(1, 2)] == 8.
    assert d[(2, 2)] == 7.
    assert d[(2, 3)] == 4.

    return True


if __name__ == '__main__':
    print(test_ChemPar())
    print(test_ReducedPar())
