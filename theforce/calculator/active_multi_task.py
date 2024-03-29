# +
# #+
import os

import numpy as np
import torch
from scipy.optimize import minimize

import theforce.distributed as distrib
from theforce.calculator.active import ActiveCalculator
from theforce.regression.multi_task import MultiTaskPotential
from theforce.util.util import abspath, date, iterable, timestamp


class MultiTaskCalculator(ActiveCalculator):
    """
    Same as ActiveCalculator, except a sequence of calculators can be passed as calculator:
        calc = MultiTaskCalculator(...,
                                   calculator=[calc0, calc1, ...],
                                   weights=[w0, w1, ...],
                                   ...)
        atoms.calc = calc
    It is required that len(calculator) = len(weights) (= tasks).
    All other arguments are the same as ActiveCalculator.

    By definition "atoms.get_potential_energy()" returns:
        e = (e0*w0 + e1*w1 + ...) / (w0 + w1 + ...)
    and "atoms.get_forces()" returns:
        f = (f0*w0 + f1*w1 + ...) / (w0 + w1 + ...)
    etc.

    Although results (energy, forces, stress) for each task individually can be obtained by:
        e = atoms.get_potential_energy() # needed!
        task = 0 # An integer in the range [0, tasks)
        task_results = atoms.calc.get_task_results(task)
        e_task = task_results['energy']
        f_task = task_results['forces']
        s_task = task_results['stress']

    Note that if tasks are identical, the calculator may become numerically unstable,
    due to low-rank structure.
    """

    def __init__(
        self,
        *args,
        weights=None,
        weights_fin=None,
        weights_sample=None,
        t_tieq=200000,
        multilogfile="multi_active.log",
        tasks_opt=True,
        niter_tasks_opt=1,
        algo="gelsd",
        k=1.0,
        d0=1.0,
        ij=None,
        **kwargs,
    ):

        self.tasks_opt = tasks_opt
        self.niter_tasks_opt = niter_tasks_opt
        self.algo = algo

        super().__init__(*args, **kwargs)

        # weights:
        assert not hasattr(self, "weights")
        if weights is None:
            weights = np.zeros(len(self._calcs))
            weights[0] = 1.0
        assert len(weights) == len(self._calcs)
        weights = np.asarray(weights)
        self.weights = weights / weights.sum()

        # final weights:
        assert len(weights_fin) == len(self._calcs)
        weights_fin = np.asarray(weights_fin)
        self.weights_fin = weights_fin / weights_fin.sum()

        self.weights_sample = weights_sample
        self.weights_init = self.weights
        self.t_tieq = t_tieq
        self.multilogfile = multilogfile

        # QMMM
        self.k = k
        self.d0 = d0
        self.ij = ij

    @property
    def tasks(self):
        return len(self._calcs)

    def get_task_results(self, task):
        result = {
            q: self.results[f"{q}_tasks"][..., task]
            for q in ["energy", "forces", "stress"]
        }
        return result

    # -----------------------------------------------

    @property
    def _calc(self):
        return self._calcs[0]

    @_calc.setter
    def _calc(self, calcs):
        if not hasattr(calcs, "__iter__"):
            calcs = [calcs]
        self._calcs = calcs

    def make_model(self, kern):
        return MultiTaskPotential(
            self.tasks, self.tasks_opt, self.niter_tasks_opt, self.algo, kern
        )

    def post_calculate(self, *args, **kwargs):

        # QMMM extension
        if self.ij is not None:
            for task in range(self.tasks):
                for pairs in self.ij:
                    # bond
                    r = self.atoms.get_distance(
                        pairs[0], pairs[1], mic=True, vector=True
                    )
                    d = np.linalg.norm(r)
                    e = self.k * (d - self.d0) ** 2
                    f = -2.0 * self.k * (d - self.d0) / d * r

                    self.results["energy"][task] += 2.0 * e
                    self.results["forces"][..., task][pairs[0]] -= f
                    self.results["forces"][..., task][pairs[1]] += f
        else:
            pass

        for q in ["energy", "forces", "stress"]:
            self.results[f"{q}_tasks"] = self.results[q]
            self.results[q] = (self.weights * self.results[q]).sum(axis=-1)
            if self.deltas:
                self.deltas[q] = (self.weights * self.deltas[q]).sum(axis=-1)

        # thermodynamic integration
        delu = ""
        if self.weights_fin is not None:
            delu = (
                self.get_task_results(0)["energy"] - self.get_task_results(1)["energy"]
            )
        self.multilog(f"{delu}  {self.weights}  {self.model.tasks_kern.view(-1)}")

        super().post_calculate(*args, **kwargs)

        # weights sampling
        if (
            self.weights_sample is not None
            and (self.step % self.weights_sample) == 0
            and self.step > 0
        ):
            self.active_sample_weights_space()

        # thermodynamic integration
        if self.weights_fin is not None and (self.step % self.t_tieq) == 0:
            self.thermo_int()

    def active_sample_weights_space(self):
        """
        A function that enforces an even sampling over the weights space w=[w0,w1,...,wn]
        """
        # enforces weights change
        while 1:
            update = np.zeros(len(self._calcs))
            update[np.random.randint(len(self._calcs))] = 1.0
            if np.dot(self.weights, update) == 0.0:
                self.weights = update
                break
        assert len(self.weights) == len(self._calcs)
        self.weights = np.asarray(self.weights)
        self.weights = self.weights / self.weights.sum()
        self.log(f"Active weights sample actived - Weights changed to w={self.weights}")

    def thermo_int(self):
        """
        To Perform the Thermodynamic Integration from w1 to w2
        \\lambda changes in time to integrate out the \\lambda in numerical grid
        w_new = (1-\\lambda)*w1 + \\lambda*w2
        """
        ti_ngrid = 10
        ti_lambda = min(round(self.step / (self.t_tieq * ti_ngrid), 1), 1.0)
        self.weights = (
            1.0 - ti_lambda
        ) * self.weights_init + ti_lambda * self.weights_fin
        self.log(f"Thermodynamics Integration in progress - Weights w={self.weights}")

    def _exact(self, copy):
        results = []
        for task, _calc in enumerate(self._calcs):
            e, f = super()._exact(copy, _calc=_calc, task=task)
            results.append((e, f))
        e, f = zip(*results)
        e = np.array(e)
        f = np.stack(f, axis=-1)
        return e, f

    def update_results(self, retain_graph=False):
        quant = ["energy", "forces", "stress"]
        local_numbers = [int(loc.number) for loc in self.atoms.loc]
        energies = self.model.predict_multitask_energies(self.cov, local_numbers)
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
        # self.log(f'Inter-task correlation: {self.model.tasks_kern}')

    def multilog(self, mssge, mode="a"):
        if self.multilogfile and self.rank == 0:
            with open(self.multilogfile, mode) as f:
                f.write("{}{} {} {}\n".format(self._logpref, date(), self.step, mssge))
                if self.stdout:
                    print("{}{} {} {}".format(self._logpref, date(), self.step, mssge))
