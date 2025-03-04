import os
import sys
import json 

import torch
#from ase.calculators.socketio import SocketClient
from ase.io import read
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator
from theforce.util.parallel import mpi_init
from theforce.util.aseutil import init_velocities
from ase import units

import numpy as np 
from ase_md_npt import NPT3
from ase_md_logger import MDLogger3
from ase.calculators.lammpslib import LAMMPSlib

process_group = mpi_init()

# Define atoms object
atoms = read('fullerene.xyz', index=0)

cmds = ["pair_style tersoff",
        "pair_coeff * * SiC.tersoff C"]
lmp_calc = LAMMPSlib(lmpcmds=cmds, log_file='test.log')

kernel_kw = {'lmax':3, 'nmax':3, 'exponent':4, 'cutoff':6.0}
calc = ActiveCalculator(covariance='pckl',
                           kernel_kw=kernel_kw,
                           calculator=lmp_calc,
                           ediff=0.04,
                           fdiff=0.04,
                           process_group=process_group,
                           pckl="model.pckl",
                           tape="model.sgpr",
                           max_data=50,
                           max_inducing=1000,
                           nbeads=1)

atoms.calc = calc


dt_fs = 0.5*units.fs
ttime = 25.0*units.fs
ptime = 100.0*units.fs
bulk_modulus = 137.0
pfactor = (ptime**2)*bulk_modulus * units.GPa
temperature_K = 300.0
temperature = temperature_K * units.kB
external_stress = 0.01 * units.GPa 

l_vel_init = True
if l_vel_init:
    init_velocities (atoms, temperature_K)

fixed_temperature = True
fixed_pressure = False

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
            trajectory='md.traj',
            logfile=None,
            append_trajectory=True,
            loginterval=10)


logger = MDLogger3 (dyn=dyn, atoms=atoms, logfile='md.dat', stress=True)
dyn.attach (logger, 2)
dyn.run (1000000)

