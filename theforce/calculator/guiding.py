from theforce.calculator.active import ActiveCalculator
from theforce.regression.guiding import GuidingPotential
import theforce.distributed as distrib
from scipy.optimize import minimize
import numpy as np
import torch
import os
from theforce.util.util import date, timestamp, abspath, iterable

class GuidingCalculator(ActiveCalculator):
    """
    GuidingCalculator actively learns the correction energy & forces between parameter force fields and ab initio calculator.
    You can set a guiding calculator (ASE force fields). The SGPR will learn the ab initio corrections. 

    Same as ActiveCalculator, except a sequence of calculators can be passed as calculator:
        calc = GuidingCalculator(...,
                                   calculator=[calc0, calc1,
                                   ...)
        atoms.calc = calc

    - calc0 is ab initio calculation. 
    - calc1 is parameter force fields. 

    All other arguments are the same as ActiveCalculator.

    """

    def __init__(self, *args, guiding_calculator=None, guidinglogfile='guiding_active.log', **kwargs):
        super().__init__(*args, **kwargs)

    def make_model(self):
        return GuidingPotential()

    def post_calculate(self, *args, **kwargs):

        for q in ['energy', 'forces', 'stress']:
            self.results[q] = (self.results[q]).sum(axis=-1)
            if self.deltas:
                self.deltas[q] = (self.deltas[q]).sum(axis=-1)

        super().post_calculate(*args, **kwargs)

