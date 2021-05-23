# +
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.calculator.meta import Meta
from theforce.util.parallel import mpi_init
from theforce.util.aseutil import init_velocities, make_cell_upper_triangular
from ase.build import bulk
from ase.md.npt import NPT
from ase import units


# define a metadynamics potential
def colvar(numbers, xyz, cell, pbc, nl):
    """
    collective variables should be generated here.
    the return value should be a 1d vector.

    nl is the neighbor list: 
    nei_i, off = nl.get_neighbors(i) # off is the ofsets due to pbc
    off = torch.from_numpy(off).type(xyz.type())
    off = (off[..., None]*cell).sum(dim=1)
    r_ij = xyz[nei_i] - xyz[i] + off
    """
    return xyz[1]-xyz[0]  # a dummy variable


meta = Meta(colvar, sigma=0.1, w=0.01)

# setup
main_calc = SocketCalculator(script='calc_emt.py')
calc = ActiveCalculator(calculator=main_calc,
                        meta=meta,  # <------------- notice here
                        process_group=mpi_init(),
                        tape='Au.sgpr')
atoms = bulk('Au', cubic=True).repeat(3*[2])
atoms.set_calculator(calc)
atoms.rattle(0.2)

# md
npt = False
tem = 1000.
stress = 0.
dt = 1.
ttime = 25*units.fs
ptime = 100*units.fs
bulk_modulus = 137.
pfactor = (ptime**2)*bulk_modulus*units.GPa
init_velocities(atoms, tem)
# make_cell_upper_triangular(atoms)
filtered = FilterDeltas(atoms)
dyn = NPT(filtered, dt*units.fs, temperature_K=tem, externalstress=stress*units.GPa,
          ttime=ttime, pfactor=pfactor if npt else None, mask=None, trajectory='md.traj',
          append_trajectory=False, loginterval=1)

# update histograms
dyn.attach(meta.update)  # <------------- notice here

# run
dyn.run(10000)
