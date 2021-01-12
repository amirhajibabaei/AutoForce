<!-- #region -->
### Introduction
This is a package for machine learning (ML) of the potential energy surface (PES)
from costly density functional theory (DFT) calculations using the Gaussian process 
regression method with emphasis on the sparse approximations.
Ab-initio calculations such as AIMD, NEB, etc can be substantially accelerated by 
fast ML models built on-the-fly.
Moreover, the ML models built with smaller size of physical systems can be applied
for simulations of larger systems which are impossible with DFT.
In principle, all the calculators supported by the atomic simulation environment 
([ASE](https://wiki.fysik.dtu.dk/ase/)) can be modeled.


### Dependencies
* <ins>Main</ins>: numpy, scipy, pytorch, atomic simulation environment (ASE), psutil
* <ins>Recommended</ins>: message passing interface (MPI enabled pytorch) for distributed computation.
* <ins>Optional</ins>: pymatgen, spglib, mendeleev, matplotlib, nglview

For enabling distributed computation with MPI, 
pytorch needs to be installed from source
(see [this](https://github.com/pytorch/pytorch)).

### Installation
Clone the source code by
```shell
git clone https://github.com/amirhajibabaei/AutoForce.git
```
Go to the source code directory and install by
```shell
pip install .
```

### Command line
For machine learning accelerated molecular dynamics
(using VASP) from the command line see 
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
<!-- #endregion -->
