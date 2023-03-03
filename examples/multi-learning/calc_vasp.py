# +
from ase.calculators.vasp import Vasp

calc = Vasp(command="mpirun -n 16 vasp_std", directory="vasp")
