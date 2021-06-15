<!-- #region -->
### ActiveCalculator
This calculator wraps the calculators defined in the 
[ASE](https://wiki.fysik.dtu.dk/ase/) package and 
generates a machine learning model on-the-fly
with the SGPR and adaptive sampling algorithms.
For instance
```python
from ase.calculators.vasp import Vasp
from theforce.calculator.active import ActiveCalculator

DFT_calc = Vasp(command='mpirun -n 12 vasp_std')
ML_calc = ActiveCalculator(calculator=DFT_calc)
```
Next, we create an `atoms` object 
(see [ase.atoms.Atoms](https://wiki.fysik.dtu.dk/ase/ase/atoms.html))
and set `ML_calc` as its calculator
```python
atoms.set_calculator(ML_calc)
```
After this, we can perform simulations such as 
[molecular dynamics](https://wiki.fysik.dtu.dk/ase/ase/md.html)
with on-the-fly machine learning.

The above script runs in parallel for DFT but 
it is serial for ML calculations.
For notes on full parallelism of both DFT and ML 
see the section **Parallelism** in the following.
Running examples can be found 
[here](https://github.com/amirhajibabaei/AutoForce/tree/master/examples).

With `ML_calc`, the ML model will be generated 
automatically on-the-fly with our arbitrary
ab-initio simulations.
For training with existing data see the section
**Training with existing data** in proceeding.

### Parameters
The following parameters can be passed to `ActiveCalculator`
```
# inputs
covariance:      either a kernel or path to a saved/pickled model
kernel_kw        e.g. {'cutoff': 6.}
calculator:      any ASE calculator or SocketCalculator
process_group:   None or the value of mpi_init()
meta:            meta energy calculator for metadynamics

# outputs
logfile:         file name for logging
pckl:            folder-name for pickling the ML model
tape:            for saving all of the model updates
test:            intervals for single point tests during MD
stdout:          if True, prints the log also to stdout

# sampling and optimization
ediff:           energy sensitivity for sampling LCEs
fdiff:           forces sensitivity for sampling DFT data
noise_f:         bias noise for forces
ioptim:          for controlling the hyper-parameter optimization
max_data:        maximum number of ab initio data
max_inducing:    maximum number of reference/inducing LCEs
veto:            for vetoing some ML updates
```

#### covariance, kernel_kw
The `covariance` keyword can be used for passing a kernel
or a saved/pickled model to the calculator
```python
from theforce.similarity.sesoap import SeSoapKernel

lmax, nmax, exponent, cutoff = 3, 3, 4, 6.
kernel = SeSoapKernel(lmax, nmax, exponent, cutoff)
ML_calc = ActiveCalculator(covariance=kernel, ...)
```

The model will be pickled (saved) after every update
in the folder given by `pckl` (default=`'model.pckl/'`).
Alternatively one can pass `pckl=None` and manually
save the model model by
```python
ML_calc.model.to_folder('model.pckl/')
```
which can be directly passed as `covariance`
in the next run
```python
ML_calc = ActiveCalculator(covariance='model.pckl/', ...)
```

The default is `covariance='pckl'` which is
equivalent to `covariance=pckl` which means
the input and output models are the same.
Thus with default setting, the model will be 
automatically saved and loaded in consecutive
runs.
If `None`, the default kernel will be used 
(see **Kernels**) with an empty initial model.
In that case `kernel_kw` can be used for 
passing some parameters (e.g. cutoff) to
the kernel instantiation.

Unless you want to use a kernel different
from the defaults, the recommended way to
instantiate the `ActiveCalculator` is to
pass `kernel_kw` (do not pass `covariance`).
Currently two implementations of the SOAP
kernels are used as defaults:
wildcard (no need to specify the atomic types)
and specified atomic types (see **Kernels**).
The latter can be much faster.
If `'species'` are given in `kernel_kw`,
the latter kernel will be used.
```python
# method 1:
kw = {'lmax': 3, 'nmax': 3, 'exponent': 4, 'cutoff': 6.} # <- any system, this is the default
ML_calc = ActiveCalculator(kernel_kw=kw)

# method 2:
kw = {'lmax': 3, 'nmax': 3, 'exponent': 4, 'cutoff': 6., species=[1, 8]} # <- specifically for water
ML_calc = ActiveCalculator(kernel_kw=kw)
```
If the model is loaded from a file,
`kernel_kw` will have no effect.

#### calculator
The main DFT calculator can which be any ASE 
calculator or a `SocketCalculator` (see *Parallelism*).
For using an existing ML model without further 
training pass `calculator=None`.

#### process_group
see **Parallelism**.

#### meta
This part is for metadynamics and it is still under development.

#### logfile, pckl, tape, test
`logfile` is used for logging.
The default name is `'active.log'`.
It can be visualized by
```python
from theforce.calculator.active import log_to_figure

fig = log_to_figure('active.log')
```

`pckl` is the folder-name for saving the pickled
model after every update step.
The default is `pckl='model.pckl'`.
For disabling this feature and manual saving
pass `pckl=None` (for less IO overhead).
For resuming the training from a saved model
we pass the folder-name instead of a kernel
as `covariance`
```python
ML_calc = ActiveCalculator(covariance='model.pckl')
```

`tape` is used for saving the most essential 
information during training (i.e. the sampled data) 
in a simple text format.
Unlike pickled models, it occupies much less memory
and can be used for rebuilding the model from scratch.
The default name is `tape='model.sgpr'`.
By design `tape`-files are never overwritten. 
These files can be used for retraining the model
with different parameters or combining several
trainig trajectories.
For instance
```python
ML_calc.include_sgpr('model.sgpr')
```

`test` can be used for exact single point
calculations during MD for on-the-fly assessment
of the ML accuracy.
The exact calculation will be saved in `'active_FP.traj'`
and the ML predictions will be saved in `'active_ML.traj'`.
A single point calculation is triggered if `test` 
steps have passed from the last one.

#### ediff, fdiff
These parameters control the sampling.
`ediff` is mainly used for sampling of the LCEs
as the inducing data for the sparse representation.
`fdiff` control the sampling of DFT data.
The default parameters should be appropriate for 
starting the simulation.
One can set `fdiff` equal to the desired accuracy
for force predictions and if the model was unable 
to reach this accuracy, `ediff` can be gradually 
decreased.
For global exploration, we recommend increasing the
accuracy gradually/iteratively rather than choosing 
small values for these parameters from the beginning.

#### noise_f
In optimization of hyper-parameters, the 
mean absolute error (MAE) of forces in fitting 
is tuned to this value: MAE$\sim$`noise_f`.
`noise_f` of 0 is also acceptable but
there is a chance for overfitting.
For instance during on-the-fly training,
choosing smaller `noise_f` may
cause more sampling of DFT data without a
significant increase in accuracy of predictions.
But the value of 0 maybe used for fitting a 
static data set without any issues.

#### ioptim
For setting the hyper-parameter optimization (HPO) frequency.
* -1 -> no HPO
* 0 -> once for every LCE/data sampled
* 1 -> only if n.o. LCE + n.o. data sampled > 0 (in 1 MD step)
* 2 -> only if new data are sampled
* i>2 -> only when n.o. sampled data is divisible by (i-1)

Frequency of HPOs decrease dramatically with increasing ioptim:
0 >> 1 >> 2. Default is ioptim = 1.

#### max_data, max_inducing
These flags can be used for increasing the speed.
The computational cost for energy and force
predictions is propodtional to the n.o. inducing LCEs.
For updating the model, the cost depends also
on the n.o. ab initio data.
If n.o. data > `max_data`, the earliest data
will be dumped.
If n.o. inducing > `max_inducing`, those LCEs
which have the maximum cumulative covariance with
the remaining LCEs will be eliminated.

#### veto
If `veto = {'forces': 8.}`, every time the predicted
forces become larger than `8.` (except the first step),
ML update is skipped.
This can be useful for algorithms such as random
structure search where many (useless) high energy
structures maybe generated during the search.
Learning these structures can be time consuming
and increases the computational cost for future force
predictions.
Using `veto` one can bypass these structures.

### Training with existing data
If some DFT data already exists, one can train a 
model simply by
```python
ML_calc.include_data(data)
```
where `data` is either a list of `Atoms` objects
or path to a (`.traj`) file.

### Parallelism
The main issue for ML parallelism is that `mpirun` 
can not be invoked twice in the same process.
Thus we need two seperate processes for ML
and DFT calculations.
Currently, this is resolved by defining the DFT 
calculator in a seperate script (e.g. `calc.py`)
```python
# calc.py
from ase.calculators.vasp import Vasp

calc = Vasp(command="mpirun -n 6 vasp_std",
            directory='vasp',
            #...
           )

# optional function
def preprocess_atoms(atoms):
    pass

# optional function
def postprocess_atoms(atoms):
    pass
```
The name `calc` should be defined in `calc.py`.
One can also define optional functions 
`preprocess_atoms` and `postprocess_atoms`.
For instance `preprocess_atoms` can be used for 
setting the initial magnetic moments for DFT 
calculations.
Then in the main python script (e.g. `md.py`)
we write
```python
# md.py
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator
from theforce.util.parallel import mpi_init

DFT_calc = SocketCalculator(script='calc.py')

ML_calc = ActiveCalculator(calculator=DFT_calc,     # main (DFT) calc
                           process_group=mpi_init() # for mpi parallelism
                           )
# ... atoms and md code
```
Then, the simulation can be started by
```sh
python -m theforce.calculator.calc_server & # A
sleep 1 # waits for the server to be up
mpirun -np 6 python md.py # B
```
The two processes `A` and `B` communicate 
through the ip `localhost` and the port `6666` by default.
ip and port can be set as optional args in command `A`
and as kwargs in `SocketCalculator`.

Commands `A` and `B` can be started on different nodes.
For instance, one can start the process `A` on `nodeA` 
and connect the nodes before starting the process `B` 
on `nodeB` by
```sh
# on nodeB
ssh -N -f -L localhost:6666:localhost:6666 nodeA
```
If the port (default=`6666`) is already occupied 
by some other process, it can be cleared by 
(before executing commands `A` or `B`)
```sh
lsof -ti:6666 | xargs kill -9
```

### Kernels

Kernels can be imported from `theforce.similarity`.
Currently the kernels in the `sesoap` module are preferred:
```python
from theforce.similarity.sesoap import SeSoapKernel, SubSeSoapKernel

lmax, nmax, exponent, cutoff = 3, 3, 4, 6.
kernel = SeSoapKernel(lmax, nmax, exponent, cutoff)
```
Using `SeSoapKernel` we don't need to worry about atomic types
but it maybe slow in some cases. 
Another option is `SubSeSoapKernel` which explicitly depends on the atomic species.
By fixing the atomic species, it can be ~10 times faster, but it uses more memory.
As an example
```python
a = SubSeSoapKernel(lmax, nmax, exponent, cutoff, 1, (1, 8))
b = SubSeSoapKernel(lmax, nmax, exponent, cutoff, 8, (1, 8))
kernel = [a, b]
```
The recommended way to use the default kernels is to pass
`kernel_kw` (see above).
Integers (positive) `lmax`, `nmax` can be decreased
for faster calculations or increased for higher accuracy.
Although the higher accuracy is not always guaranteed
(we haven't tested).
<!-- #endregion -->
