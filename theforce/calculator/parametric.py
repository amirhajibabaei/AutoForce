#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.nn import Module
from theforce.util.util import iterable
from theforce.math.func import Negative, Positive, Real, Pow, Param, I
from theforce.math.cutoff import PolyCut
import itertools


class ParametricPotential(Module):

    def __init__(self):
        super().__init__()
        self.params = []

    def forward(self, atoms_or_loc, forces=False, enable_grad=True):
        with torch.set_grad_enabled(enable_grad):
            if forces:
                f = 0
            e = 0
            for loc in iterable(atoms_or_loc):
                _e = self.calculate(loc, forces=forces)
                if forces:
                    _e, _f = _e
                    f = f + _f
                e = e + _e
            if forces:
                return e, f
            else:
                return e

    def __add__(self, other):
        return AddedPotentials(self, other)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            raise RuntimeError('This is not ok: {} + {}'.format(self, other))

    def __repr__(self):
        return self.state

    @property
    def unique_params(self):
        params = []
        ids = []
        for param in self.params:
            if id(param) not in ids:
                params.append(param)
                ids.append(id(param))
        return params

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)


class AddedPotentials(ParametricPotential):

    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.params = a.params + b.params

    def forward(self, atoms_or_loc, forces=False, enable_grad=True):
        a = self.a(atoms_or_loc, forces=forces, enable_grad=enable_grad)
        b = self.b(atoms_or_loc, forces=forces, enable_grad=enable_grad)
        if forces:
            a, aa = a
            b, bb = b
            return a + b, aa + bb
        else:
            return a + b

    @property
    def state_args(self):
        return '{}, {}'.format(self.a.state, self.b.state)

    @property
    def state(self):
        return '{} + {}'.format(self.a.state, self.b.state)


class PairPot(ParametricPotential):

    def __init__(self, a, b, radial):
        super().__init__()
        self.a = a
        self.b = b
        self.radial = radial
        try:
            self.params += radial.params
        except AttributeError:
            pass

    def calculate(self, loc, forces=False):
        loc.select(self.a, self.b, bothways=False)
        d = loc.r.pow(2).sum(dim=-1, keepdim=True).sqrt()
        e = self.radial(d, grad=forces)
        if forces:
            e, g = e
            f = -g*loc.r/d
            f = torch.zeros(loc.natoms, 3).index_add(
                0, loc.j, f).index_add(0, loc.i, -f)
            return e.sum(), f
        else:
            return e.sum()

    @property
    def state_args(self):
        return '{}, {}, {}'.format(self.a, self.b, self.radial.state)


def get_coulomb_terms(numbers, cutoff, setting={}, default_order=0.01):
    """
    If a setting is passed, it should be a dictionary in the form of 
    {atomic_number: c} where c contains the constraint and optionally 
    the initial value. Acceptable constraints are 
    '+': positive
    '-': negative
    'r': real (no constraints)
    'f': fixed
    ------------------------------------------------------
    examples:
    c = '+' -> "positive" constraint
    c = ('+', 1.) -> "positive" constraint, initial value 1
    c = ('r', 1.) -> no constraint, initial value 1
    c = ('f', 1.) -> fix the charge to value 1
    """
    # initialize charges
    charges = {}
    for a in numbers:
        try:
            # constraint and initial value
            c = setting[a]
            if type(c) == str:
                if c == '+' or c == 'r':
                    ini = default_order
                elif c == '-':
                    ini = -default_order
                elif c == 'f':
                    raise RuntimeError('f (=fixed) constraint needs a value')
                else:
                    raise RuntimeError('unknown constraint {}'.format(c))
            else:
                c, ini = c
            # class of constraint
            if c == '+':
                _cls = Positive
                rg = True
            elif c == '-':
                _cls = Negative
                rg = True
            elif c == 'r':
                _cls = Real
                rg = True
            elif c == 'f':
                _cls = Real
                rg = False
            else:
                raise RuntimeError('unknown constraint {}'.format(c))
            # create charge
            charges[a] = Param(_cls, ini, 'q_{}'.format(a), rg=rg)
        except KeyError:
            charges[a] = Param(Real, default_order, 'q_{}'.format(a), rg=True)
    # create terms
    pairs = ([(a, b) for a, b in itertools.combinations(numbers, 2)] +
             [(a, a) for a in numbers])

    terms = sum([PairPot(*pair, PolyCut(cutoff)*charges[pair[0]]*charges[pair[1]]*Pow(n=-1))
                 for pair in pairs])
    return terms


def get_lj_terms(numbers, cutoff, default_order=0.01):
    # create terms
    pairs = ([(a, b) for a, b in itertools.combinations(numbers, 2)] +
             [(a, a) for a in numbers])
    A = {pair: Param(Positive, default_order, 'A_{}_{}'.format(*pair))
         for pair in pairs}
    B = {pair: Param(Positive, default_order, 'B_{}_{}'.format(*pair))
         for pair in pairs}
    terms = sum([PairPot(*pair, PolyCut(cutoff)*(A[pair]*Pow(n=-12) - B[pair]*Pow(n=-6)))
                 for pair in pairs])
    return terms


def load_parametric_potential(state):
    return eval(state)


def test():
    from theforce.descriptor.atoms import TorchAtoms
    from theforce.math.radial import RepulsiveCore
    torch.set_default_tensor_type(torch.DoubleTensor)

    V = PairPot(55, 55, RepulsiveCore()) + PairPot(55, 55, RepulsiveCore())
    a = TorchAtoms(positions=[(0, 0, 0), (2, 0, 0), (0, 2, 0)],
                   numbers=[55, 55, 55], cell=[10, 10, 10], pbc=False)
    a.update(cutoff=5., posgrad=True)
    e, f = V(a, forces=True)
    e.backward()
    print(a.xyz.grad.allclose(-f))
    print(V.state)
    print(sum([PairPot(55, 55, RepulsiveCore()),
               PairPot(55, 55, RepulsiveCore())]))


if __name__ == '__main__':
    test()

