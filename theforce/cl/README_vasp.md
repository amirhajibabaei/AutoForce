<!-- #region -->
### VASP
Here we describe the steps required for the machine learning accelerated
molecular dynamics (MLMD) using VASP from the command line.
For this, we create the usual files required for a VASP calculation
with few minor modifications and extensions.

#### POSCAR and KPOINTS
Create these files as usual.

#### POTCAR
`POTCAR` will be created automatically if the environment variable
`VASP_PP_PATH` is properly set.
For more information see [this](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html).
In this case one can control which potcars are used by creating a
file called `SETUPS`.
For example a `SETUPS` containing the following will use `Li_sv`.
```
Li = _sv
```

#### INCAR
Set the tags in `INCAR` as usual while adhering to the following.

First, `INCAR` should be prepared only for single-point
energy and force calculations.
We shouldn't set any tags related to ionic steps in `INCAR`
because `ase.md` will drive the dynamics.
Only tags related to the electronic steps should be set.

Second, if necessary, we set the initial magnetic moments not in `INCAR`,
but in a new file named `IMAG`.
For example a line like
```
28 = 2.
```
in `IMAG` sets the initial magnetic moment of 2. for all Ni atoms.

#### ARGS (MLMD related)
All parameters related to ML (input and output models, target accuracy, etc)
and task specification (MD, relaxation, etc) should be set
in the `ARGS` file (see
[theforce/cl/README.md](https://github.com/amirhajibabaei/AutoForce/tree/master/theforce/cl)).
Make sure to have the following line in `ARGS`
```
calculator = 'VASP'
```

#### Run
Lets assume that 20 cores are available.
We split these cores to 12 for VASP and 8 for ML.
Then in a file named `COMMAND` we write
```sh
mpirun -n 12 vasp_std
```
After this, the simulation can be started with
the following script
```sh
python -m theforce.calculator.calc_server &
sleep 1 # waits 1 sec for the server to be set
#
### choose one of following depending on you task
# mpirun -n 8 python -m theforce.cl.init_model  # for model initialization
# mpirun -n 8 python -m theforce.cl.relax       # for structure relaxation
mpirun -n 8 python -m theforce.cl.md            # for ML accelerated MD
```
This will try to read the initial coordinates
from `POSCAR` and will write the final coordinates
in `CONTCAR` for convenience.
Optionally other file names can be passed as input
and ouput using `-i input-name` and `-o output-name`
command line arguments.
<!-- #endregion -->
