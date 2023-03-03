### LAMMPS

#### Coupled with on-the-fly ML-IAP using VASP
In this example, LAMMPS is used for MD where
force field is generated on-the-fly with AutoForce
using the data obtained from DFT calculations with VASP.
Note the following:
* Atomic numbers, which are needed for DFT, should be specified by (`in.lammps`):
```
#AutoForce          atomic_numbers = {1:6, 2:1}
```
* The forces are calculated via fix-external mechanism (`in.lammps`):
```
fix                 AutoForce all external pf/callback 1 1
fix_modify          AutoForce energy yes virial yes
```
* DFT calculator and ML params are set in `ARGS`.
* `COMMAND` contains the command for running VASP.
* `INCAR` is for VASP settings.
* `run.sh` shows how to run the simulation.

Assuming that 20 CPUs are availabe in the machine,
4 CPUs are used for LAMMPS+AutoForce (see `run.sh`)
and 16 for VASP (see `COMMAND`).

This algorithm allows us to use diverse MD options
which are available in LAMMPS (e.g. metadynamics, etc.).

#### Using pre-trained force-fields

For using pre-trained force-fields,
only LAMMPS-specific input files are needed.
Additionally one should copy a force-field
(e.g. `model.pckl/`) to the working directory.
Then:
```
mpirun -np 20 python -m theforce.cl.lmp -i in.lammps
```
This time all 20 CPUs are used for LAMMPS+AutoForce
since no on-the-fly training with VASP is performed.
This is also possible just by setting `calculator = None`
in `ARGS`.
