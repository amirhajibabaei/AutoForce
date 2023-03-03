# +
import re

import spglib
from ase.spacegroup import Spacegroup


def get_cell(atoms):
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    cell = (lattice, positions, numbers)
    return cell


def tag_sites(atoms):
    name = spglib.get_spacegroup(get_cell(atoms))
    number = int(re.search(r"\((.*?)\)", name).group(1))
    spg = Spacegroup(number)
    sites = spg.tag_sites(atoms.get_scaled_positions())
    return sites
