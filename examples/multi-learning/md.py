# +
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.util.parallel import mpi_init
from theforce.util.aseutil import init_velocities
from ase.build import bulk
from ase.md.npt import NPT
from ase import units

process_group = mpi_init()
common = dict(ediff=0.01, fdiff=0.01, process_group=process_group)

# driver calc
_calc_1 = SocketCalculator(script='calc_emt.py')
calc_1 = ActiveCalculator(calculator=_calc_1,
                          logfile='active_1.log',
                          pckl='model_1.pckl/',
                          tape='model_1.sgpr',
                          **common)

# observer calc
_calc_2 = SocketCalculator(script='calc_vasp.py')
calc_2 = ActiveCalculator(calculator=_calc_2,
                          logfile='active_2.log',
                          pckl='model_2.pckl/',
                          tape='model_2.sgpr',
                          **common)


atoms = bulk('Au', cubic=True).repeat(3*[3])
atoms.calc = calc_1
atoms.rattle(0.1)
init_velocities(atoms, 1000*units.kB)
filtered = FilterDeltas(atoms)
dyn = NPT(filtered, 2*units.fs, temperature_K=1000.,
          externalstress=0., trajectory='md.traj')


def observe(a=atoms):
    calc_2.calculate(atoms)


dyn.attach(observe)
dyn.run(1000)
