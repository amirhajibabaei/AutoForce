# +
import numpy as np
from ase.atoms import Atoms
from ase.optimize import LBFGS

from theforce.calculator.active import ActiveCalculator, kcal_mol
from theforce.calculator.socketcalc import SocketCalculator
from theforce.util.flake import generate_random_cluster
from theforce.util.parallel import mpi_init

# Initiate MPI and sync np.random in all processes:
# this should be called before random cluster generation
# so that all processes generate the identical cluster.
process_group = mpi_init()


# random cluster generation
ngold = 20
min_dist = 2.5
positions = generate_random_cluster(ngold, min_dist)
atoms = Atoms(numbers=ngold * [79], positions=positions)
atoms.center(vacuum=5.0)
atoms.pbc = True


# Ab initio calculator; for now we just use EMT instead of vasp
# abinitio = SocketCalculator(script='calc_vasp.py')
abinitio = SocketCalculator(script="calc_emt.py")


# ML calculator
active_kwargs = {
    "calculator": abinitio,
    "ediff": 1.0 * kcal_mol,  # decrease for more accuracy but lower speed
    "fdiff": 1.0 * kcal_mol,  # decrease for more accuracy but lower speed
    "kernel_kw": {"cutoff": 6.0, "lmax": 3, "nmax": 3},
    # 'kernel_kw': {'cutoff': 6., 'lmax': 3, 'nmax': 3, 'species': [79]}, # <- faster
    # 'veto': {'forces': 8.}  # for vetoing ML updates for very high energy structures
    "process_group": process_group,
}
calc = ActiveCalculator(**active_kwargs)
atoms.calc = calc


# master rank: for IO
master = calc.rank == 0


# relax
maxforce = 0.01
dyn = LBFGS(atoms, trajectory="relax.traj", master=master)
dyn.run(fmax=maxforce)
# For history dependent algorithms such as LBFGS, if ML
# updates becomes problematic, dyn.run can be replaced by:
#
# for _ in dyn.irun(fmax=maxforce):
#    if calc.updated:
#        dyn.initialize()    # clears history


# Getting closer to the local minima by forceful updating
# the model; this will cause at least 2 more ab initio
# calculations (one of which is needed anyway to determine
# the proximity to the actual minima).
while True:
    if calc.update_data(try_fake=False):
        calc.update(data=False)
        calc.results.clear()
        dyn.initialize()
        dyn.run(fmax=maxforce)
    else:
        break


# Calculate exact energy & forces for the final coordinates.
# The optimized coordinates and ab initio energy & forces
# will be written to "active_FP.traj".
energy, forces = calc._test()
f_rms = np.sqrt(np.mean(forces**2))
f_max = abs(forces).max()
report = f"""
    relaxation result:
    energy:      {energy}
    force (rms): {f_rms}
    force (max): {f_max}
"""
if master:
    print(report)


# save the final structure
if master:
    # atoms.write('CONTCAR')
    atoms.write("optimized.xyz", format="extxyz")
