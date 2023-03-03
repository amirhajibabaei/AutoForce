from ase.build import bulk

atoms = bulk("Au", cubic=True).repeat(3 * [3])
atoms.write("atoms.xyz")
