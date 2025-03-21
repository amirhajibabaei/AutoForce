"""
This module defines an ASE interface to LAMMPS
"""
from ase import Atoms
from ase.calculators.lammpslib import LAMMPSlib
from ase.io import read

if __name__ == "__main__":
    #from ase.io import read 
    import sys
    
    #atoms = read ('w256.xyz')
    atoms = read ('fullerene.xyz')
    
    cmds = ["pair_style airebo       6 0 0",
            "pair_coeff * * CH.airebo   C"]
    atoms.calc = LAMMPSlib(lmpcmds=cmds, log_file='test_log')
    #volbox = atoms.get_volume()
    enr = atoms.get_potential_energy()
    frc = atoms.get_forces ()
    print ('enr', enr)
    print ('frc', frc[:6])
    
    
