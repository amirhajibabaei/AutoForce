# +
from autoforce.functions import FiniteRange
import torch


def test_FiniteRange() -> bool:
    r = FiniteRange(0., 1.)
    x = torch.tensor([0.0, 0.5, 1.0])
    test = r.function(r.inverse(x)).allclose(x)
    return test


if __name__ == '__main__':
    a = test_FiniteRange()
    print(a)
