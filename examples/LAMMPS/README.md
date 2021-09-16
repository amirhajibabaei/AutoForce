### LAMMPS coupled with on-the-fly ML-IAP and VASP

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
* `INCAR` is for VASP settings.
* `run.sh` shows how to run the simulation.

This algorithm allows us to use diverse MD options
which are available in LAMMPS (e.g. metadynamics, etc.).
