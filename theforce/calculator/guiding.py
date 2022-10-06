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

    @property
    def tasks(self):
        return len(self._calcs)

    def get_task_results(self, task):
        result = {
            q: self.results[f'{q}_tasks'][..., task]
            for q in ['energy', 'forces', 'stress']
        }
        return result

    # -----------------------------------------------

    @property
    def _calc(self):
        return self._calcs[0]

    @_calc.setter
    def _calc(self, calcs):
        if not hasattr(calcs, '__iter__'):
            calcs = [calcs]
        self._calcs = calcs

    def make_model(self):
        return GuidingPotential()

    def post_calculate(self, *args, **kwargs):

        for q in ['energy', 'forces', 'stress']:
            self.results[q] = (self.results[q]).sum(axis=-1)
            if self.deltas:
                self.deltas[q] = (self.deltas[q]).sum(axis=-1)

        super().post_calculate(*args, **kwargs)

    def calculate(self, *args, **kwargs):
        #calculate FF
        _calc=self._calcs[1]
        copy = self.atoms.copy()
        e_ff, f_ff = super()._exact(copy, _calc=_calc)
        #self._calcs[1].calculate(self, atoms, properties, system_changes)
        #obtain guided calculation results
        super().calculate(*args, **kwargs)
        #add the correction - how to access energy and forces?
        self.results['energy'] += e_ff
        self.results['forces'] += f_ff

    def _exact(self, copy):
        _calc=self._calcs[0]
        results = []
        e, f = super()._exact(copy, _calc=_calc)
        results.append((e, f))
        e, f = zip(*results)
        #learn the correction
        e = np.array(e)-e_ff
        f = np.stack(f, axis=-1)-f_ff
        return e, f

    def update_results(self, retain_graph=False):
        quant = ['energy', 'forces', 'stress']
        local_numbers = [int(loc.number) for loc in self.atoms.loc]
        energies = self.model.predict_multitask_energies(
            self.cov, local_numbers)
        results = []
        for e in energies:
            self.reduce(e, retain_graph=True)
            results.append([self.results[q] for q in quant])
        e, f, s = zip(*results)
        e = np.array(e)
        f = np.stack(f, axis=-1)
        s = np.stack(s, axis=-1)
        for q, v in zip(quant, [e, f, s]):
            self.results[q] = v

    def guidinglog(self, mssge, mode='a'):
        if self.guidinglogfile and self.rank == 0:
            with open(self.guidinglogfile, mode) as f:
                f.write('{}{} {} {}\n'.format(
                    self._logpref, date(), self.step, mssge))
                if self.stdout:
                    print('{}{} {} {}'.format(
                        self._logpref, date(), self.step, mssge))
