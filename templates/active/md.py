# +
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.similarity.sesoap import SeSoapKernel
from theforce.util.parallel import mpi_init
from theforce.util.aseutil import init_velocities, make_cell_upper_triangular
from ase.build import bulk
from ase.md.npt import NPT
from ase import units

# A. the main dft calculator (see calc_emt.py), we use the fast emt for getting started
main_calc = SocketCalculator(script='calc_emt.py') # or calc_vasp.py

# B. the kernel and the active learning calculator
lmax, nmax, exponent, cutoff = 3, 3, 4, 6.
kern = SeSoapKernel(lmax, nmax, exponent, cutoff)
calc = ActiveCalculator(covariance=kern,          # kern, if None the default is used
                        calculator=main_calc,     # main (DFT) calc
                        ediff=0.01, fdiff=0.05,   # control params for accuracy/speed tradeoff
                        process_group=mpi_init(), # for mpi parallelism
                        pckl='model.pckl/',       # for continuous pickling (saving) the model
                        tape='Au.sgpr',           # for saving the training data step by step
                        test=100,                 # test 100 steps after last dft (None for no test)
                        )
# note that, all of the above have default values.
# for a minimal  example, you only need to set
# main_calc and process_group.
# the model will be saved on-the-fly in a folder
# given by the pckl argument which can be used for
# resuming the training by covariance='model.pckl/'.
# if pckl=None, the model is not saved on the fly.
# one can manually save the model after training by
#    calc.model.to_folder('model.pckl/')
# in restarting, if main_calc=None, the model
# will be loaded but it will not be updated.

# C. define the system and set its calculator
atoms = bulk('Au', cubic=True).repeat(3*[3])
atoms.set_calculator(calc)

# D. intitial state with all-forces=0 is problematic
#    with an empty model at the beginning.
#    this can be ignored but there is a possibility
#    for things to go wrong.
atoms.rattle(0.1)

# E. define md
# 1. set the time-step dt smaller than you would
#    with direct AIMD, at least if you are starting
#    with an empty model. if the model is mature,
#    you can set dt large, even larger than dt
#    for AIMD.
# 2. we wrap atoms with "FilterDeltas" for damping
#    of the sudden change of forces uppon upating the
#    potential. often this is not seriesly consequential
#    and can be ignored.
# 3. set "npt = True" if you with to do NPT.
#    otherwise it will do NVT.
# 4. you need to make the cell upper triangular
#    (if it already isn't) before using ase.md.npt.NPT.
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

# F. run md
dyn.run(1000)

# G. save model manually (necessary only if pckl=None above)
# calc.model.to_folder('model.pckl')

# H. restart
#    calc = ActiveCalculator(covariance='model.pckl/', ...)

# I. using the saved tapes (.sgpr files)
# these files can be used for faster retraining the model e. g.
# with different parameters, etc.
#    calc.include_tape('model.sgpr')
# this mechanism can also be used for high level parallelism.
# one can train independent models at different domains and
# combine them by including their respective tapes.

# J. training with existing data:
#    calc.include_data(data)
# where data is either a list of atoms objects or
# path to a traj file e. g. 'data.traj'.
