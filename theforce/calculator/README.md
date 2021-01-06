<!-- #region -->
### ActiveCalculator
This calculator wraps the calculators defined in the 
[ASE](https://wiki.fysik.dtu.dk/ase/) package and 
generates a machine learning model on-the-fly
with the SGPR and adaptive sampling algorithms.
For instance
```python
from ase.calculators.vasp import Vasp2
from theforce.calculator.active import ActiveCalculator

DFT_calc = Vasp2(command='mpirun -n 12 vasp_std')
ML_calc = ActiveCalculator(calculator=DFT_calc)
```
Next, we create an `atoms` object 
(see [ase.atoms.Atoms](https://wiki.fysik.dtu.dk/ase/ase/atoms.html))
and set `ML_calc` as its calculator
```python
atoms.set_calculator(ML_calc)
```
After this, we can perform simulations such as 
[molecular dynamics](https://wiki.fysik.dtu.dk/ase/ase/md.html).

The above script runs in parallel for DFT but 
it is serial for ML calculations.
For notes on full parallelism of both DFT and ML 
see the section **Parallelism** in the following.
Running examples can be found 
[here](https://github.com/amirhajibabaei/AutoForce/tree/master/templates).

### Parameters
The following parameters can be passed to `ActiveCalculator`
```
covariance:      either a kernel or path to a saved/pickled model
calculator:      any ASE calculator or SocketCalculator
process_group:   None or the value of mpi_init()
ediff:           energy sensitivity for sampling LCEs
fdiff:           forces sensitivity for sampling DFT data
coveps:          for skipping ML update
covdiff:         usefull only in special cases
meta:            meta energy calculator for metadynamics
logfile:         file name for logging
tape:            for checkpointing the ML model
```
#### covariance
This parameter can be used for passing a kernel
or a saved/pickled model to the calculator
```python
from theforce.similarity.sesoap import SeSoapKernel

lmax, nmax, exponent, cutoff = 3, 3, 4, 6.
kernel = SeSoapKernel(lmax, nmax, exponent, cutoff)
ML_calc = ActiveCalculator(covariance=kernel, ...)
```
If not given, the default kernel will be used.

After the training is finished, one can save a pickled
model by
```python
ML_calc.model.to_folder('model/')
```
which can be directly passed as `covariance`
in the next run
```python
ML_calc = ActiveCalculator(covariance='model/', ...)
```

#### calculator
The main DFT calculator can which be any ASE 
calculator or a `SocketCalculator`.
For using an existing ML model without further 
training pass `calculator=None`.

#### ediff, fdiff, coveps, covdiff
These parameters control the sampling.
The most important parameter is `fdiff` which 
ideally should determine the accuracy (stdev) of
the resulting model for force predictions.
`ediff` is mainly used for sampling of the LCEs
as the inducing data for sparse representation.
`ediff` should be set according to `fdiff`:
first we set the desired `fdiff` and if 
the model is unable to fit the data with 
this accuracy we can lower `ediff` for
denser sampling of LCEs.
Smaller values for these parameters should
increase the accuracy but also increases the
computational cost.

`coveps` can be used for skipping ML updates
and speed up.
If `covloss < coveps` the ML update is skipped.
`covloss` is monitored in the `logfile`.
`covdiff` is used in the initial stages of 
training where the model has less than two
reference LCEs.
In this case if `covloss > covdiff`, it will
be assumed that the model is blind and DFT 
calculation will be triggered.

#### meta
This part is for metadynamics and it is still under development.

#### logfile, tape
`logfile` is used for logging.
The default name is `active.log`.
It can be visualized by
```python
from theforce.calculator.active import log_to_figure

fig = log_to_figure('active.log')
```

`tape` is used for saving the most essential 
information during training (i.e. the sampled data) 
in a simple text format.
Unlike pickled models, it occupies very little memory
and can be used for rebuilding the model from scratch.
The default name is `tape='model.sgpr'`.
By design these files are never overwritten. 
Thus if it exists in the working directory it will 
be loaded automatically and the new updates will be
appended to it.
A pickled model can be created from an existing `tape` by
```python
ML_calc = ActiveCalculator(tape='model.sgpr', ...)
ML_calc.model.to_folder('model/')
```

### Parallelism
The main issue for ML parallelism is that `mpirun` 
can not be invoked twice in the same process.
Thus we need to seperate the processes for ML 
and DFT calculations.
Currently, this is resolved by defining the DFT 
calculator in a seperate script (e.g. `calc.py`)
```python
# calc.py
from ase.calculators.vasp import Vasp2

calc = Vasp2(command="mpirun -n 6 vasp_std", 
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

calc = ActiveCalculator(calculator=DFT_calc,     # main (DFT) calc
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
<!-- #endregion -->
