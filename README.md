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

### Installation
Clone the source code by
```shell
git clone https://github.com/amirhajibabaei/AutoForce.git
```
Go to the source code directory and install by
```shell
pip install .
```

### Usage
It wraps ASE calculators:

```python
from theforce.calculator.active import ActiveCalculator

# atoms = see ASE docs
# main_calc = see ASE calculators
# kernel = see the proceeding

calc = ActiveCalculator(kernel, calculator=main_calc)
atoms.set_calculator(calc)

# proceed with the desired calculations
# ...

# save and load models
calc.model.to_folder('model')
calc = ActiveCalculator('model', calculator=main_calc)
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
