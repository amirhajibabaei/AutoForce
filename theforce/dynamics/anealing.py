#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase import units
from theforce.dynamics.leapfrog import Leapfrog
from theforce.regression.gppotential import PosteriorPotentialFromFolder
import numpy as np
import warnings


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
                           pressure=None, stress_equilibration=5, rescale_cell=1.01, eps='random',
                           algorithm='fastfast', name='model', overwrite=True, traj='tempering.traj',
                           logfile='leapfrog.log'):
    """
    pressure (hydrostatic): 
        defined in units of Pascal and is equal to -(trace of stress tensor)/3 
    eps:
        if 'random', strain *= a random number [0, 1)
        if a positive float, strain *= 1-e^(-|dp/p|/eps) i.e. eps ~ relative p fluctuations
        else, no action
    """
    assert rescale_velocities > 1 and rescale_cell > 1
    if pressure is not None:
        warnings.warn('rescaling cell is not robust!')

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
            if pressure is not None:
                spu, e, T, s = dyn.run_updates(stress_equilibration)
                t += spu*stress_equilibration*dt
                p = -s[:3].mean() / units.Pascal
                # figure out strain
                dp = p - pressure
                strain = (rescale_cell if dp > 0 else 1./rescale_cell) - 1
                if eps is 'random':
                    strain *= np.random.uniform()
                elif type(eps) == float and eps > 0 and abs(p) > 0:
                    strain *= 1 - np.exp(-np.abs(dp/p)/eps)
                # apply strain
                dyn.strain_atoms(np.eye(3)*strain)

        if k == stages-1:
            dyn.model.to_folder(name, info='temperature: {}'.format(T),
                                overwrite=overwrite)
        else:
            dyn.model.to_folder('{}_{}'.format(name, k), info='temperature: {}'.format(T),
                                overwrite=overwrite)
    return dyn.get_atoms(), dyn.model


def train_pes_by_tempering(atoms, gp, cutoff, ttime, calculator=None, model=None, dt=2., ediff=0.01, volatile=None,
                           target_temperature=1000., stages=1, equilibration=5, rescale_velocities=1.05,
                           pressure=None, stress_equilibration=5, rescale_cell=1.01, randomize=True,
                           algorithm='fastfast', name='model', overwrite=True, traj='tempering.traj',
                           logfile='leapfrog.log'):
    assert rescale_velocities > 1 and rescale_cell > 1

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
            if pressure is not None:
                spu, e, T, s = dyn.run_updates(stress_equilibration)
                t += spu*stress_equilibration*dt
                p = -s[:3].mean() / units.Pascal
                # figure out factor
                dp = p - pressure
                factor = (rescale_cell if dp > 0 else 1./rescale_cell)
                if randomize:
                    factor = 1 + np.random.uniform(0, 1)*(factor-1)
                # apply rescaling
                dyn.rescale_cell(factor)

        if k == stages-1:
            dyn.model.to_folder(name, info='temperature: {}'.format(T),
                                overwrite=overwrite)
        else:
            dyn.model.to_folder('{}_{}'.format(name, k), info='temperature: {}'.format(T),
                                overwrite=overwrite)
    return dyn.get_atoms(), dyn.model

