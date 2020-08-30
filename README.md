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
Main:
* numpy 
* scipy
* pytorch
* atomic simulation environment (ASE)
* psutil

Recommended: 
* message passing interface (MPI enabled pytorch) for distributed computation.

Optional:
* pymatgen 
* spglib 
* mendeleev 
* matplotlib
* nglview


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
```

---

Kernels can be imported from `theforce.similarity`.
`UniversalSoapKernel` is the easiest to implement:
```python
from theforce.similarity.universal import UniversalSoapKernel

lmax, nmax, exponent, cutoff = 3, 3, 4, 6.
kernel = UniversalSoapKernel(lmax, nmax, exponent, cutoff)
```
but it maybe slow in some cases. 

---

Another option is `HeterogeneousSoapKernel` which segregates atomic species, 
is faster (~10 times), but it uses more memory.
A utility function is present for creating the relevent list of kernels given 
the atomic species present in the system:
```python
from theforce.run.fly import default_kernel

# numbers = [1, 8] for water
# cutoff, etc can be given as kwargs
kernel = default_kernel(numbers) 
```
<!-- #endregion -->
