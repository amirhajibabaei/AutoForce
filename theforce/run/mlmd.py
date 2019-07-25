
# coding: utf-8

# In[ ]:


from ase.md.verlet import VelocityVerlet
from theforce.calculator.posterior import PosteriorVarianceCalculator
from theforce.run.pes import potential_energy_surface
from theforce.descriptor.atoms import TorchAtoms, AtomsData, LocalsData, sample_atoms
import numpy as np


def mlmd(ini_atoms, cutoff, au, dt, tolerance=0.1, pair=True, soap=True, max_data=20, max_steps=100,
         itrain=30, retrain=0, pes=potential_energy_surface):
    """ 
    ML-assisted-MD: a calculator must be attached to ini_atoms 
    """

    dftcalc = ini_atoms.get_calculator()

    # run a short MD to gather some (dft) data
    atoms = TorchAtoms(ase_atoms=ini_atoms.copy())
    atoms.set_velocities(ini_atoms.get_velocities())
    atoms.set_calculator(dftcalc)
    dyn = VelocityVerlet(atoms, dt=dt, trajectory='md.traj', logfile='md.log')
    md_step = max_data//2
    dyn.run(md_step)
    ndft = md_step

    # train a potential
    data = AtomsData(traj='md.traj', cutoff=cutoff)
    V = pes(data, cutoff=cutoff, nlocals=-1, atomic_unit=au, pairkernel=pair,
            soapkernel=soap, train=itrain, test=True)
    atoms.update(cutoff=cutoff, descriptors=V.gp.kern.kernels)
    mlcalc = PosteriorVarianceCalculator(V)
    atoms.set_calculator(mlcalc)

    # long MD
    while md_step < max_steps:

        md_step += 1

        forces = atoms.get_forces()
        var = atoms.calc.results['forces_var']
        tol = np.sqrt(var.max(axis=1))

        if (tol > tolerance).any():

            _forces = forces
            _var = var

            # new dft
            ndft += 1
            print('|............... new dft calculation (total={})'.format(ndft))
            tmp = atoms.copy()
            tmp.set_calculator(dftcalc)
            true_forces = tmp.get_forces()
            data += [TorchAtoms(ase_atoms=tmp, cutoff=cutoff)]

            # new model
            print('|............... new regression')
            if len(data) > max_data:
                del data.X[0]
            V = pes(data, nlocals=-1, train=retrain, append_log=True)
            mlcalc = PosteriorVarianceCalculator(V)
            atoms.set_calculator(mlcalc)

            forces = atoms.get_forces()
            var = atoms.calc.results['forces_var']

            # report
            _err_pred = np.sqrt(_var).max()
            _err = np.abs(_forces - true_forces).max()
            err_pred = np.sqrt(var).max()
            err = np.abs(forces - true_forces).max()
            print(
                '|............... : old max-error: predicted={}, true={}'.format(_err_pred, _err))
            print(
                '|............... : new max-error: predicted={}, true={}'.format(err_pred, err))
            arrays = np.concatenate(
                [true_forces, _forces, forces, _var, var], axis=1)
            with open('forces_var.txt', 'ab') as report:
                np.savetxt(report, arrays)

        print(md_step, '_')
        dyn.run(1)


def example():
    from ase.calculators.lj import LennardJones
    from theforce.util.flake import Hex
    from ase.io.trajectory import Trajectory
    from ase.optimize import BFGSLineSearch
    from ase.io import Trajectory

    # initial state + dft calculator
    ini_atoms = TorchAtoms(positions=Hex().array(), cutoff=3.0)
    dftcalc = LennardJones()
    ini_atoms.set_calculator(dftcalc)
    BFGSLineSearch(ini_atoms).run(fmax=0.01)
    vel = np.random.uniform(-1., 1., size=ini_atoms.positions.shape)*1.
    vel -= vel.mean(axis=0)
    ini_atoms.set_velocities(vel)

    # use a pretrained model by writing it to the checkpoint
    pre = """GaussianProcessPotential([PairKernel(RBF(signal=3.2251566545458794, lengthscale=tensor([0.1040])), 
    0, 0, factor=PolyCut(3.0, n=2))], White(signal=0.1064043798026091, requires_grad=True))""".replace('\n', '')
    with open('gp.chp', 'w') as chp:
        chp.write(pre)

    # run(md)
    mlmd(ini_atoms, 3.0, 0.5, 0.05, tolerance=0.1, soap=False, itrain=[0])

    # recalculate all with the actual calculator and compare
    traj = Trajectory('md.traj')
    energies = []
    forces = []
    dum = 0
    for atoms in traj:
        dum += 1
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        dftcalc.calculate(atoms)
        ee = dftcalc.results['energy']
        ff = dftcalc.results['forces']
        energies += [(e, ee)]
        forces += [(f.reshape(-1), ff.reshape(-1))]

    import pylab as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].scatter(*zip(*energies))
    axes[0].set_xlabel('ml')
    axes[0].set_ylabel('dft')
    a, b = (np.concatenate(v) for v in zip(*forces))
    axes[1].scatter(a, b)
    axes[1].set_xlabel('ml')
    axes[1].set_ylabel('dft')
    fig.tight_layout()
    fig.text(0.2, 0.8, 'energy')
    fig.text(0.7, 0.8, 'forces')


if __name__ == '__main__':
    import os
    os.system('rm -f log.txt gp.chp md.log md.traj forces_var.txt')
    example()

