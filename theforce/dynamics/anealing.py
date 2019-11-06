#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase import units
from theforce.dynamics.leapfrog import Leapfrog
from theforce.regression.gppotential import PosteriorPotentialFromFolder
import numpy as np


def learn_pes_by_anealing(atoms, gp, cutoff, calculator=None, model=None, dt=2., ediff=0.01, volatile=None,
                          target_temperature=1000., stages=1, equilibration=5, rescale_velocities=1.05,
                          algorithm='fastfast', name='model', overwrite=True, traj='anealing.traj',
                          logfile='leapfrog.log'):
    assert rescale_velocities > 1

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
                   ediff=ediff, volatile=volatile, algorithm=algorithm, logfile=logfile)

    # initial equilibration
    while dyn.volatile():
        _, e, t, s = dyn.run_updates(1)
    _, e, t, s = dyn.run_updates(equilibration)

    temperatures = np.linspace(t, target_temperature, stages+1)[1:]
    heating = t < target_temperature
    cooling = not heating
    for k, target_t in enumerate(temperatures):
        print('stage: {}, temperature: {}, target temperature: {}, ({})'.format(
            k, t, target_t, 'heating' if heating else 'cooling'))
        while (heating and t < target_t) or (cooling and t > target_t):
            dyn.rescale_velocities(
                rescale_velocities if heating else 1./rescale_velocities)
            _, e, t, s = dyn.run_updates(equilibration)
        if k == stages-1:
            dyn.model.to_folder(name, info='temperature: {}'.format(t),
                                overwrite=overwrite)
        else:
            dyn.model.to_folder('{}_{}'.format(name, k), info='temperature: {}'.format(t),
                                overwrite=overwrite)
    return dyn.get_atoms(), dyn.model


def learn_pes_by_tempering(atoms, gp, cutoff, ttime, calculator=None, model=None, dt=2., ediff=0.01, volatile=None,
                           target_temperature=1000., stages=1, equilibration=5, rescale_velocities=1.05,
                           algorithm='fastfast', name='model', overwrite=True, traj='tempering.traj',
                           logfile='leapfrog.log'):
    assert rescale_velocities > 1

    if model is not None:
        if type(model) == str:
            model = PosteriorPotentialFromFolder(model)
        if gp is None:
            gp = model.gp

    if atoms.get_velocities() is None:
        t = target_temperature
        MaxwellBoltzmannDistribution(atoms, t*units.kB)
        Stationary(atoms)
        ZeroRotation(atoms)

    dyn = VelocityVerlet(atoms, dt*units.fs, trajectory=traj)
    dyn = Leapfrog(dyn, gp, cutoff, calculator=calculator, model=model,
                   ediff=ediff, volatile=volatile, algorithm=algorithm, logfile=logfile)

    t = 0
    T = '{} (instant)'.format(atoms.get_temperature())
    checkpoints = np.linspace(0, ttime, stages+1)[1:]
    for k, target_t in enumerate(checkpoints):
        print('stage: {}, time: {}, target time: {}, (temperature={})'.format(
            k, t, target_t, T))
        while t < target_t:
            spu, e, T, s = dyn.run_updates(equilibration)
            t += spu*equilibration*dt
            dyn.rescale_velocities(
                rescale_velocities if T < target_temperature else 1./rescale_velocities)

        if k == stages-1:
            dyn.model.to_folder(name, info='temperature: {}'.format(T),
                                overwrite=overwrite)
        else:
            dyn.model.to_folder('{}_{}'.format(name, k), info='temperature: {}'.format(T),
                                overwrite=overwrite)
    return dyn.get_atoms(), dyn.model

