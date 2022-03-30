# +
from autoforce.core import Bijection
from torch import Tensor
import torch


class FiniteRange(Bijection):

    def __init__(self, a: float, b: float) -> None:

        if a > b:
            raise RuntimeError('Lower bound is larger '
                               'than upper bound!'
                               )

        self.a = torch.as_tensor(a)
        self.b = torch.as_tensor(b)
        self.l = b - a

    def __repr__(self):
        return f'{self.__class__.__name__}({self.a}, {self.b})'

    def forward(self, x: Tensor) -> Tensor:
        x = torch.as_tensor(x)
        y = self.a + self.l*torch.special.expit(x)
        return y

    def inverse(self, y: Tensor) -> Tensor:
        y = torch.as_tensor(y)
        x = (y-self.a)/self.l
        return torch.special.logit(x)


def test_FiniteRange():
    r = FiniteRange(0., 1.)
    x = torch.tensor([0.0, 0.5, 1.0])
    test = r.forward(r.inverse(x)).allclose(x)
    return test


if __name__ == '__main__':
    a = test_FiniteRange()
    print(a)
