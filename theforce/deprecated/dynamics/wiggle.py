import numpy as np
from ase.md.verlet import VelocityVerlet
import math
import warnings


class Oscillator:

    def __init__(self, amplitude=0.05, period=1000):
        self.amp = amplitude
        self.w = 2*math.pi/period
        self.t = 0.

    @property
    def f(self):
        return 1 + self.amp * np.sin(self.w*self.t)

    @property
    def df(self):
        return self.amp * self.w * np.cos(self.w*self.t)

    @property
    def d2f(self):
        return - self.amp * self.w**2 * np.sin(self.w*self.t)

    @property
    def a(self):
        return self.df/self.f

    @property
    def b(self):
        return self.d2f/self.f

    def step(self):
        self.t += 1


class Wiggle(VelocityVerlet):

    def __init__(self, *args, amplitude=0.05, period=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.osc = Oscillator(amplitude=amplitude, period=period)
        warnings.warn("This dynamics is not stable, it is only experimental")

    def step(self, f=None):
        # algorithm is the same as ase.md.verlet.VelocityVerlet
        # with a few lines added
        atoms = self.atoms
        if f is None:
            f = atoms.get_forces()
        r = atoms.get_positions()
        p = atoms.get_momenta()
        masses = atoms.get_masses()[:, np.newaxis]
        c = (atoms.get_scaled_positions()-0.5)@atoms.get_cell()  # added
        f += (self.osc.b*c + 2*self.osc.a *
              (p/masses - self.osc.a*c))*masses  # added
        p += 0.5 * self.dt * f
        atoms.set_positions(r + self.dt * p / masses)
        gamma = 1 + self.dt*(self.osc.a + 0.5*self.dt*self.osc.b)  # added
        atoms.set_cell(atoms.get_cell()*gamma, scale_atoms=False)  # added
        if atoms.constraints:
            p = (atoms.get_positions() - r) * masses / self.dt
        atoms.set_momenta(p, apply_constraint=False)
        f = atoms.get_forces(md=True)
        self.osc.step()
        c = (atoms.get_scaled_positions()-0.5)@atoms.get_cell()  # added
        f += (self.osc.b*c + 2*self.osc.a *
              (p/masses - self.osc.a*c))*masses  # added
        atoms.set_momenta(atoms.get_momenta() + 0.5 * self.dt * f)
        return f

