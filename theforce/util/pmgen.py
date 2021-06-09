# +
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation


def standard_cell(prim, **kwargs):
    """kwargs: symprec, angle_tolerance, ..."""
    trans = ConventionalCellTransformation(**kwargs)
    return AseAtomsAdaptor.get_atoms(trans.apply_transformation(
        AseAtomsAdaptor.get_structure(prim)))
