# +
import numpy as np
import torch
from torch.nn import Module, Parameter


class PairPotential(Module):

    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        if a == b:
            self.double_count = 2
        else:
            self.double_count = 1

    def forward(self, a, b, r):
        if a != self.a:
            return 0.0
        else:
            m = torch.as_tensor(b == self.b)
            return self.energy(r[m]) / self.double_count

    def energy(self):
        raise NotImplementedError(
            'energy shoud be implemented in a child class')


class LJ(PairPotential):

    def __init__(self, a, b, eps=1.0, sigma=1.0, rc=3.0):
        super().__init__(a, b)
        self.rc = rc
        self.eps = eps
        self.sigma = sigma
        self.e0 = (sigma/rc)**12 - (sigma/rc)**6

    def energy(self, r):
        d2 = (r**2).sum(dim=-1)
        x = self.sigma**2/d2[d2 < self.rc**2]
        e = 4*(self.eps*(x**6 - x**3 - self.e0)).sum()
        return e
