
import sys 
import numpy as np

from ase.build import bulk
from ase.optimize import BFGS 
from ase.io import read 
from theforce.util.parallel import mpi_init 
from theforce.util.aseutil import init_velocities 
from ase import units
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active_bcm import BCMActiveCalculator

from ase_md_npt import NPT3
from ase_md_logger import MDLogger3



process_group = mpi_init()
kernel_kw = {'lmax': 3, 'nmax':3, 'exponent':4, 'cutoff':6.0} 

main_calc = SocketCalculator(script="emt_calc.py") 

bcm_calc = BCMActiveCalculator (
    covariance='pckl',
    kernel_kw=kernel_kw,
    calculator=main_calc,
    ediff=0.02,
    fdiff=0.02,
    process_group=process_group,
    logfile='bcm_active.log',
    pckl="model_bcm.pckl",
    max_data=50,
    max_inducing=1000,
    test=100
)

atoms = bulk("Au", cubic=True).repeat(3 * [3])
atoms.set_calculator(bcm_calc)
atoms.rattle(0.1)

if False:
    opt = BFGS (atoms, logfile='bfgs_opt.out')
    opt.run (fmax=1.0)

dt_fs = 0.5*units.fs
ttime = 25.0*units.fs
ptime = 100.0*units.fs
bulk_modulus = 137.0
pfactor = (ptime**2)*bulk_modulus * units.GPa
temperature_K = 300.0
temperature = temperature_K * units.kB
external_stress = 0.01 * units.GPa 

init_velocities (atoms, temperature_K)

fixed_temperature = True
fixed_pressure = True 

if not fixed_temperature:
    ttime=None

if not fixed_pressure:
    pfactor = None 

anisotropic = False

dyn = NPT3 (atoms,
            dt_fs,
            temperature=temperature, 
            externalstress=external_stress,
            ttime=ttime,
            pfactor=pfactor,
            anisotropic=anisotropic,
            trajectory='md_bcm.traj',
            logfile=None,
            append_trajectory=True,
            loginterval=200)

logger = MDLogger3 (dyn=dyn, atoms=atoms, logfile='md_bcm.dat', stress=True)
dyn.attach (logger, 2)
dyn.run (2000)
