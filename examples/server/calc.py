# +
from ase.calculators.vasp import Vasp

calc = Vasp(
    command="mpirun -n 6 vasp_std",
    directory="vasp",
    ispin=2,
    kspacing=0.5,
)


def preprocess_atoms(atoms):
    atoms.set_initial_magnetic_moments(len(atoms) * [1.0])
