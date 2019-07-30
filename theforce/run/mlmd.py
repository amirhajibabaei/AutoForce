
# coding: utf-8

# In[ ]:


from ase.md.verlet import VelocityVerlet
from theforce.calculator.posterior import PosteriorVarianceCalculator
from theforce.run.pes import potential_energy_surface
from theforce.descriptor.atoms import TorchAtoms, AtomsData, LocalsData, sample_atoms
from theforce.util.util import iterable
import numpy as np


def mlmd(ini_atoms, cutoff, au, dt, tolerance=0.1, pair=True, soap=True, ndata=10, max_steps=100,
         itrain=10*[5], retrain=5*[5], retrain_every=100, pes=potential_energy_surface):
    """ 
    ML-assisted-MD: a calculator must be attached to ini_atoms.
    Rules of thumb:
    Initial training (itrain) is crucial for correct approximation 
    of variances.
    Hyper-parameters are sensitive to nlocals=len(inducing) thus 
    if you don't want to retrain gp every time the data is updated, 
    at least keep nlocals fixed.
    """

    dftcalc = ini_atoms.get_calculator()

    # run a short MD to gather some (dft) data
    atoms = TorchAtoms(ase_atoms=ini_atoms.copy())
    atoms.set_velocities(ini_atoms.get_velocities())
    atoms.set_calculator(dftcalc)
    dyn = VelocityVerlet(atoms, dt=dt, trajectory='md.traj', logfile='md.log')
    md_step = ndata
    dyn.run(md_step)
    ndft = md_step

    # train a potential
    data = AtomsData(traj='md.traj', cutoff=cutoff)
    inducing = data.to_locals()
    V = pes(data=data, inducing=inducing, cutoff=cutoff, atomic_unit=au, pairkernel=pair,
            soapkernel=soap, train=itrain, test=True, caching=True)
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

            # new dft calculation
            ndft += 1
            print('|............... new dft calculation (total={})'.format(ndft))
            tmp = atoms.copy()
            tmp.set_calculator(dftcalc)
            true_forces = tmp.get_forces()

            # add new information to data
            new_data = AtomsData(X=[TorchAtoms(ase_atoms=tmp)])
            new_data.update(
                cutoff=cutoff, descriptors=atoms.calc.potential.gp.kern.kernels)
            new_locals = new_data.to_locals()
            new_locals.stage(descriptors=atoms.calc.potential.gp.kern.kernels)
            data += new_data
            inducing += new_locals  # TODO: importance sampling

            # remove old(est) information
            del data.X[0]
            del inducing.X[:len(new_locals)]  # TODO: importance sampling

            # retrain
            if ndft % retrain_every == 0:
                print('|............... : retraining for {} steps'.format(retrain))
                for steps in iterable(retrain):
                    atoms.calc.potential.train(data, inducing=inducing,
                                               steps=steps,  cov_loss=False)
                    atoms.calc.potential.gp.to_file(
                        'gp.chp', flag='ndft={}'.format(ndft))

            # update model
            print('|............... new regression')
            atoms.calc.potential.set_data(data, inducing, use_caching=True)

            # new forces
            atoms.calc.results.clear()
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
    print('finished {} steps, used dftcalc only {} times'.format(md_step, ndft))


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
    # (alternatively set, for example, itrain=10*[5] in mlmd)
    pre = """GaussianProcessPotential([PairKernel(RBF(signal=3.2251566545458794, lengthscale=tensor([0.1040])),
    0, 0, factor=PolyCut(3.0, n=2))], White(signal=0.1064043798026091, requires_grad=True))""".replace('\n', '')
    with open('gp.chp', 'w') as chp:
        chp.write(pre)

    # run(md)
    mlmd(ini_atoms, 3.0, 0.5, 0.03, tolerance=0.18, max_steps=100,
         soap=False, itrain=10*[3], retrain_every=5, retrain=5)

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

