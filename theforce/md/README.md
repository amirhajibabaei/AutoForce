<!-- #region -->
### MLMD from commandline
Here we describe the steps required for the machine learning accelerated 
molecular dynamics (MLMD) using VASP from the command line.
For this, we create the usual files required for a VASP calculation 
with few minor modifications and extensions.

#### POSCAR and KPOINTS
Create these files as usual.

#### POTCAR
`POTCAR` can also be generated as usual, if one doesn't want it 
to be generated automatically.

`POTCAR` will be created automatically if no `POTCAR` is present
in the working directory and the environment variable 
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

First, we shouldn't set any tags related to ionic steps in `INCAR`
because `ase.md.npt` will drive the dynamics.
Only tags related to the electronic steps should be set.

Second, if necessary, we set the initial magnetic moments not in `INCAR`,
but in a new file named `IMAG`.
For example a line like 
```
28 = 2.
```
in `IMAG` sets the initial magnetic moment of 2. for all Ni atoms.

#### Parameters for MD 
The parameters for MD are set in a file named `MD`.
The following tags are available
```
dt:           time-step in fs (default=1. or 0.25 if H is present)
tem:          temperature in Kelvin (default=300.)
picos:        pico-seconds for md (default=100)
bulk_modulus: bulk_modulus for NPT simulations. if None (default), NVT is performed
stress:       external stress (GPa) for NPT (default=0.)
mask:         see ase.npt.NPT (default=None)
tape:         checkpoint for the ML potential (default='model.sgpr')
trajectory:   traj file name (default='md.traj')
loginterval:  for traj file (default=1)
append:       append to traj file (default=False)
rattle:       rattle atoms at the initial step (default=0.05)
```
All of the above tags have default values which will be overridden
with the settings in `MD`. 
A minimal `MD` file could contain the following
```
tem = 300.
picos = 20
```

A practical issue may rise if the MD simulation starts 
with an empty model and from an initial forces very 
close to zero.
Although this is not a issue in most of the cases, 
sometimes the active learning algorithm may fail.
For this, we have introduced the `rattle` tag 
which disturbs the atoms at the initial state.
The default value is `rattle=0.05`.

The default timestep is `dt=1.` femtosecond 
(`dt=0.25` if hydrogen is present) which is 
intentionally smaller than typical `dt` for AIMD
because updating the model on-the-fly causes 
discontinuities in the forces.
Smaller `dt` increases the stability in these cases.
If the ML model is mature, larger `dt` can be used
(even larger than AIMD).

If `tape` is not given, the ML updates will be 
saved in a file called `model.sgpr`.
If this file is present when the simulation starts,
it will be loaded automatically.
This file can be used for checkpointing.

If `bulk_modulus` is given, NPT simulation
will be performed (with cell fluctuations).
It is recommended to first perform a NVT
simulation and then use the result `model.sgpr`
as the starting potential for the NPT simulation.
NPT simulations are more vulnerable to sudden ML updates.
If the model is mature, this is not an issue.

#### Run
Lets assume that 20 cores are available.
We split these cores to 12 for VASP and 8 for ML.
Then in a file named `COMMAND` we write
```sh
mpirun -n 12 vasp_std
```
After this, the simulation can be started with the following script
```sh
python -m theforce.calculator.calc_server &
sleep 1 # waits 1 sec for the server to be set
mpirun -n 8 theforce.md.vasp
```

#### Outputs
The trajectory is saved in `md.traj` by default.
This file can be processed using the `ase.io` module.
The log for ML is written in `active.log` which 
can be visualized (converted to `active.pdf`) by
```sh
python -m theforce.calculator.active active.log
```
ML updates are saved in the tape (default=`model.sgpr`)
which can be used as the input model for the next
simulation.
<!-- #endregion -->
