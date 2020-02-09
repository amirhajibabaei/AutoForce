
import torch
from theforce.math.func import Func

class PairCut(Func):

    def __init__(self, cutoff):
        super().__init__()
        self.rc = cutoff

    @property
    def state_args(self):
        return '{}'.format(self.rc)

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

    def forward(self, d, grad=True):
        step = torch.where(d < self.rc, torch.ones_like(d),
                           torch.zeros_like(d))
        f = self.func_and_grad(d, grad=grad)
        if grad:
            f, g = f
            return step*f, step*g
        else:
            return step*f

    def func_and_grad(self, d, grad=True):
        raise NotImplementedError('func_and_grad')


class PolyCut(PairCut):

    def __init__(self, cutoff, n=2):
        super().__init__(cutoff)
        self.n = n

    def func_and_grad(self, d, grad=True):
        if grad:
            return (1.-d/self.rc)**self.n, -self.n*(1.-d/self.rc)**(self.n-1)/self.rc
        else:
            return (1.-d/self.rc)**self.n

    @property
    def state_args(self):
        return super().state_args + ', n={}'.format(self.n)


def test():
    cut = PolyCut(0.5)
    d = torch.rand(10, 1)
    r, dr = cut(d)


if __name__ == '__main__':
    test()

