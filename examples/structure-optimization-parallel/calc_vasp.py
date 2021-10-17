# +
from ase.calculators.vasp import Vasp


calc = Vasp(command="mpirun -n 4 vasp_std",
            directory='vasp',
            ispin=2,
            )


def preprocess_atoms(atoms):
    # optional: can be deleted
    # this can be defined for setting the parameters
    # that can not be set when defining the calculator.
    #
    # for instance:
    # atoms.set_initial_magnetic_moments(len(atoms)*[1.])
    pass


def postprocess_atoms(atoms):
    # optional
    pass
