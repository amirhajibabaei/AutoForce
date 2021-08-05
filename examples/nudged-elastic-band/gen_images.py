# Copied (and modified) from ASE tutorial at:
# https://wiki.fysik.dtu.dk/ase/tutorials/neb/diffusion.html
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.neb import NEB
from ase.io import Trajectory

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
# print(mask)
slab.set_constraint(FixAtoms(mask=mask))

# Use EMT potential:
slab.calc = EMT()

# Initial state:
qn = QuasiNewton(slab)
qn.run(fmax=0.05)
initial = slab.copy()

# Final state:
slab[-1].x += slab.get_cell()[0, 0] / 2
qn = QuasiNewton(slab)
qn.run(fmax=0.05)
final = slab.copy()

# constraints
constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial])

# generate images
images = [initial]
for i in range(3):
    image = initial.copy()
    image.set_constraint(constraint)
    images.append(image)
images.append(final)
neb = NEB(images)
neb.interpolate()

# write images
traj = Trajectory('images.traj', 'w')
for atoms in images:
    traj.write(atoms)
