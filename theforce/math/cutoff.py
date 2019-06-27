
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module


class PairCut(Module):

    def __init__(self, cutoff):
        super().__init__()
        self.rc = cutoff

    @property
    def state_args(self):
        return '{}'.format(self.rc)

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

    def forward(self, d):
        step = torch.where(d < self.rc, torch.ones_like(d),
                           torch.zeros_like(d))
        f, g = self.func_and_grad(d)
        return step*f, step*g

    def func_and_grad(self, d):
        raise NotImplementedError('func_and_grad')


class PolyCut(PairCut):

    def __init__(self, cutoff, n=2):
        super().__init__(cutoff)
        self.n = n

    def func_and_grad(self, d):
        return (1.-d/self.rc)**self.n, -self.n*(1.-d/self.rc)**(self.n-1)/self.rc

    @property
    def state_args(self):
        return super().state_args + ', n={}'.format(self.n)


def test():
    cut = PolyCut(0.5)
    d = torch.rand(10, 1)
    r, dr = cut(d)


if __name__ == '__main__':
    test()

