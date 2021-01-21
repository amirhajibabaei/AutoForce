<!-- #region -->
### Machine learning from command line
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

#### Parameters for ML (file=`ARGS`)
The parameters for ML should be provided in a file named `ARGS`. 
This group of parameters are used for setting up the machine
learning calculator and are the same for most ML tasks.
The following tags are available
```
# inputs
covariance:      None, a kernal, folder-name for loading a pickled model (default=None)
calculator:      None or 'VASP' (default=None)

# outputs
logfile:         file name for logging (default='active.log')
pckl:            folder-name for pickling the ML model (default='model.pckl')
tape:            for saving all of the model updates (default='model.sgpr')
test:            single-point testing intervals (default=None)

# sampling and optimization
ediff:     (eV)  energy sensitivity for sampling LCEs (default ~ 2 kcal/mol)
ediff_tot: (eV)  total energy sensitivity for sampling DFT data (default ~ 5 kcal/mol)
fdiff:    (eV/A) forces sensitivity for sampling DFT data (default ~ 2 kcal/mol)
noise_e:   (ev)  bias noise for total energies (default=ediff_tot)
noise_f:  (ev/A) bias noise for forces (default=fdiff)
```
These parameters are explained in detail in
[theforce/calculator/README.md](https://github.com/amirhajibabaei/AutoForce/tree/master/theforce/calculator).
The only difference is that here we specify the 
calculator by its name (e.g. `calculator='VASP'`).
For a brief recap read the following.

By default the ML model will be saved in a folder
specified by `pckl` after every update.
For loading the saved model in next simulations
put the following line in `ARGS` after the first 
simulation has ended
```
covariance = 'model.pckl'
```
Thus `covariance` is the input model and `pckl`
is the output model.

After sufficient training, one might want to use
the result ML potential for fast simulations 
without further training.
For this, all you need to do is to not specify any 
`calculator` in `ARGS` or set `calculator=None`.

The other paramteres which control the sampling
and accuracy should be gradually tuned to get the
desired accuracy.

#### Parameters for MD (file=`ARGS`)
The parameters for MD are also set in the `ARGS` file.
The following tags are available
```
dt:           time-step in fs (default=1. or 0.25 if H is present)
tem:          temperature in Kelvin (default=300.)
picos:        pico-seconds for md (default=100)
bulk_modulus: bulk_modulus for NPT simulations. if None (default), NVT is performed
stress:       external stress (GPa) for NPT (default=0.)
mask:         see ase.npt.NPT (default=None)
trajectory:   traj file name (default='md.traj')
loginterval:  for traj file (default=1)
append:       append to traj file (default=False)
rattle:       rattle atoms at the initial step (default=0.0)
```
All of the above tags have default values which will be overridden
with the settings in `ARGS`. 
A minimal `ARGS` file could contain the following
```
calculator = 'VASP'
tem = 300.
picos = 20
```

The default timestep is `dt=1.` femtosecond 
(`dt=0.25` if hydrogen is present) which is 
intentionally smaller than typical `dt` for AIMD
because updating the model on-the-fly causes 
discontinuities in the forces.
Smaller `dt` increases the stability in these cases.
If the ML model is mature, larger `dt` can be used
(even larger than AIMD).

If `bulk_modulus` is given, NPT simulation
will be performed (with cell fluctuations).
It is recommended to first perform a NVT
simulation and then use the result `pckl`
as the starting potential for the NPT simulation.
NPT simulations are more vulnerable to sudden ML updates.
If the model is mature, this is not an issue.

A practical issue may rise if the MD simulation starts 
with an empty model and a state with forces close to zero.
Although this is not a issue in most cases, 
sometimes the active learning algorithm may fail.
For this, we have introduced the `rattle` tag 
which disturbs the atoms at the initial state.
The default value is `rattle=0.0`.

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
mpirun -n 8 python -m theforce.cl.md
```
This will try to read the initial coordinates
from `POSCAR` and will write the final coordinates
in `CONTCAR` for convenience.
Optionally other file names can be passed as input
and ouput using `-i input-name` and `-o output-name`
command line arguments.

#### Outputs
The trajectory is saved in `trajectory='md.traj'` by default.
This file can be processed using the `ase.io` module.
The log for ML is written in `logfile='active.log'` which 
can be visualized (converted to `active.pdf`) by
```sh
python -m theforce.calculator.active active.log
```
The resulting ML model is saved in the `pckl` 
(default=`'model.pckl'`) which can be used as the 
input model for the next simulation 
(by `covariance='model.pckl'` in `ARGS`).
ML updates are saved in the file specified by `tape`
(default=`'model.sgpr'`).
This file can be used for retraining with different
parameters, etc.

#### Post-training
After the training is finished, we can perform MD
using only the ML potential (without further updates) simply by
```sh
mpirun -n 20 python -m theforce.cl.md
```
provided that `ARGS` has the following entries
```
covariance = `model.pckl` # path to the saved model
calculator = None         # not necessary because this is the default
```
This time all 20 cores are used for ML and we can simulate
much bigger systems.

### Training with existing data
A utility command is provided for training with existing DFT data.
For this, set the appropriate parameters for ML in `ARGS`.
Then refer to the following patterns.
```sh
mpirun -n 20 python -m theforce.cl.train -i data.traj           # read all data
mpirun -n 20 python -m theforce.cl.train -i data.traj -r 0:10:2 # read only 0, 2, ..., 10
mpirun -n 20 python -m theforce.cl.train -i OUTCAR              # read all data
mpirun -n 20 python -m theforce.cl.train -i OUTCAR -r :-1:      # read only the last one
mpirun -n 20 python -m theforce.cl.train -i OUTCAR-1 OUTCAR-2   # read two OUTCAR files
mpirun -n 20 python -m theforce.cl.train -i */OUTCAR            # all OUTCAR files in subfolders
mpirun -n 20 python -m theforce.cl.train -i model.sgpr          # use the tape from another training
```
Make sure to track the input and output models using the
`covariance` and `pckl` tags in the `ARGS` file.
Similarly, for testing
```sh
mpirun -n 20 python -m theforce.cl.test -i dft.traj -o ml.traj # optionally -r ::
python -m theforce.regression.scores ml.traj dft.traj          # prints a description of errors   
```
<!-- #endregion -->
