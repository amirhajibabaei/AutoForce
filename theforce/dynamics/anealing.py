#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase import units
from theforce.dynamics.leapfrog import Leapfrog
from theforce.regression.gppotential import PosteriorPotentialFromFolder
import numpy as np


def learn_pes_by_anealing(atoms, gp, cutoff, calculator=None, model=None, dt=2., ediff=0.01, initialization=None,
                          target_temperature=1000., stages=1, equilibration=5, rescale_velocities=1.05,
                          algorithm=None, name='model', overwrite=True, traj='anealing.traj'):

    if model is not None:
        if type(model) == str:
            model = PosteriorPotentialFromFolder(model)
        if gp is None:
            gp = model.gp

    if atoms.get_velocities() is None:
        t = target_temperature / stages
        MaxwellBoltzmannDistribution(atoms, t*units.kB)
        Stationary(atoms)
        ZeroRotation(atoms)

    dyn = VelocityVerlet(atoms, dt*units.fs, trajectory=traj)
    dyn = Leapfrog(dyn, gp, cutoff, calculator=calculator, model=model,
                   ediff=ediff, init=initialization, algorithm=algorithm)

    _, ei, ti = dyn.run_updates(equilibration)
    temperatures = np.linspace(ti, target_temperature, stages+1)[1:]
    for k, target_t in enumerate(temperatures):
        print('stage: {} target temperature: {}'.format(k, target_t))
        t = 0
        while t < target_t:
            dyn.rescale_velocities(rescale_velocities)
            _, e, t = dyn.run_updates(equilibration)
        if k == stages-1:
            dyn.model.to_folder(name, info='temperature: {}'.format(t),
                                overwrite=overwrite)
        else:
            dyn.model.to_folder('{}_{}'.format(name, k), info='temperature: {}'.format(t),
                                overwrite=overwrite)
    return dyn.get_atoms(), dyn.model

