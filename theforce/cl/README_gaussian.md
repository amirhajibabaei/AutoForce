<!-- #region -->
### Gaussian
For the machine learning accelerated molecular dynamics (MLMD) 
using the Gaussian software from the command line, we generate
the input file for Gaussian (default=`Gaussian.gjf`) as usual.
This file contains the parameters for ab initio calculations
as well as the inital coordinates of atoms.
The type of calculation should be set to single-point 
energy and force calculation (use the `Force` keyword).
As an example:
```
#Force hf

water molecule

0 1
O                 0.0000000000        0.0000000000        0.1192620000
H                 0.0000000000        0.7632390000       -0.4770470000
H                 0.0000000000       -0.7632390000       -0.4770470000
```
During MLMD this file will be copied for all ab initio calculations, 
only the coordinates of atoms will be changed.

#### ARGS (MLMD related)
All parameters related to ML (input and output models, target accuracy, etc) 
and task specification (MD, relaxation, etc) should be set
in the `ARGS` file (see
[theforce/cl/README.md](https://github.com/amirhajibabaei/AutoForce/tree/master/theforce/cl)).
Make sure to have the following line in `ARGS`
```
calculator = 'Gaussian'
```

#### Run
Lets assume that 20 cores are available.
We split these cores to 12 for Gaussian and 8 for ML.
Specification of 12 cores for Gaussian should be done
in its input file.
Optionally in a file named `COMMAND` we may write 
```sh
Gaussian_exe < input.gjf > output.log
```
If `COMMAND` is not present, the following is assumed
```sh
(g16|g09|g03) < Gaussian.gjf > Gaussian.log
```
After this, the simulation can be started with 
the following script
```sh
python -m theforce.calculator.calc_server &
sleep 1 # waits 1 sec for the server to be set
#
### choose one of following depending on you task
# mpirun -n 8 python -m theforce.cl.init_model -i Gaussian.gjf   # for model initialization
# mpirun -n 8 python -m theforce.cl.relax -i Gaussian.gjf        # for structure relaxation
mpirun -n 8 python -m theforce.cl.md -i Gaussian.gjf             # for ML accelerated MD
```
Optionally other file names can be passed as input
and ouput using `-i input-name` and `-o output-name`
command line arguments.
Gaussian calculations are carried out in a directory
named `gaussian_wd`.
<!-- #endregion -->
