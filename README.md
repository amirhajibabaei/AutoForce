<!-- #region -->
### Introduction
This is a package for machine learning (ML) of the potential energy surface (PES)
from costly ab initio calculations using the sparse Gaussian process regression
(SGPR) algorithm.
Ab initio calculations such as structure relaxation, AIMD, NEB, etc. can be
substantially accelerated by fast ML models built on-the-fly.
Moreover, the ML models built with smaller size of physical systems can be applied
for simulations of larger systems which are impossible with direct ab initio methods.
In principle, all the calculators supported by the atomic simulation environment
([ASE](https://wiki.fysik.dtu.dk/ase/)) can be modeled.

### Dependencies
* <ins>required</ins>: `numpy`, `scipy`, `pytorch`, `ase`, `mpi`
* <ins>conditional</ins>: `mpi4py` (see below)
* <ins>optional</ins>: `pymatgen`, `spglib`, `mendeleev`, `matplotlib`, `nglview`, `psutil`

`mpi4py` is only required if `pytorch` is not directly linked
with `mpi` (i.e. `torch.distributed.is_mpi_available() == False`).
Note that coupling `pytorch` with `mpi` needs compilation from the source.
This package is regularly synced with the latest
versions of ASE and pytorch.
Additional setting maybe needed for linking
the ab initio calculators (VASP, GAUSSIAN, etc.)
with ASE ([see this](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#module-ase.calculators)).

### Installation
Clone the source code by
```shell
git clone https://github.com/amirhajibabaei/AutoForce.git
```
Go to the source code directory and install by
```shell
pip install .
```

### Command line interface
For machine learning accelerated molecular dynamics,
structure relaxation, etc (using VASP, GAUSSIAN, etc.) from the command line see
[theforce/cl/README.md](https://github.com/amirhajibabaei/AutoForce/tree/master/theforce/cl).

### Python API
It wraps ASE calculators:
```python
from theforce.calculator.active import ActiveCalculator

# atoms = see ASE docs
# main_calc = see ASE calculators
# kernel = see the proceeding

calc = ActiveCalculator(calculator=main_calc)
atoms.set_calculator(calc)

# proceed with the desired calculations
# ...
```
For detailed information see 
[theforce/calculator/README.md](https://github.com/amirhajibabaei/AutoForce/tree/master/theforce/calculator).


### Examples
For usage examples, see the [examples/](https://github.com/amirhajibabaei/AutoForce/tree/master/examples) folder.

### Practical notes

#### On-the-fly ML
* <ins>Ab initio calculations</ins>:
The ab-initio calculators should be used only for single-point energy and force calculations.
If on-the-fly ML fails, first and foremost, check if the underlying ab initio
calculations (for the electronic structure) do converge.
* <ins>ML models</ins>:
The default settings are such that ML models are automatically saved and loaded
in consecutive simulations. Thus check if the proper model is present in the working
directory.
* <ins>Initial structure for MD</ins>:
Starting MD with a relaxed strucure (forces=0) is not advised.
Either manually disturb the initial structure or use the `rattle`
mechanism.
* <ins>Structure optimization</ins>:
Many structure relaxation algorithms depend on the forces history.
With on-the-fly ML, every time the model is updated, forces suddenly change.
The force discontinuity, if too large, may corrupt the optimizer.
This can be avoided by reseting the optimizer history or training
a preliminary model before relaxation.

#### Scalability
* <ins>Distributed computing with MPI</ins>:
The algorithm can use at most N (=number of atoms in the system) processes
during MD. Using more processes can only speed-up the ML updates.
* <ins>CUDA</ins>:
Currently no GPU acceleration is implemented.
* <ins>Species</ins>:
Presence of more atomic species makes the simulation slower (often exponentially).

### Citation
```
@article{PhysRevB.103.214102,
  title = {Sparse Gaussian process potentials: Application to lithium diffusivity in superionic conducting solid electrolytes},
  author = {Hajibabaei, Amir and Myung, Chang Woo and Kim, Kwang S.},
  journal = {Phys. Rev. B},
  volume = {103},
  issue = {21},
  pages = {214102},
  numpages = {7},
  year = {2021},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.103.214102},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.103.214102}
}
```

<br/><br/><br/>
![alt text](https://github.com/amirhajibabaei/AutoForce/blob/master/docs/sources/UNIST.jpg?raw=True)
<!-- #endregion -->
