"""Logging for molecular dynamics."""

import weakref

import ase.units as units
from ase.parallel import world
from ase.utils import IOContext


class MDLogger3(IOContext):
    """Class for logging molecular dynamics simulations.

    Parameters:
    dyn:           The dynamics.  Only a weak reference is kept.

    atoms:         The atoms.

    logfile:       File name or open file, "-" meaning standard output.

    stress=False:  Include stress in log.

    peratom=False: Write energies per atom.

    mode="a":      How the file is opened if logfile is a filename.
    """

    def __init__(self, dyn, atoms, logfile, header=True, stress=False,
                 peratom=False, mode="a"):
        if hasattr(dyn, "get_time"):
            self.dyn = weakref.proxy(dyn)
        else:
            self.dyn = None
        self.atoms = atoms
        global_natoms = len(atoms) #atoms.get_global_number_of_atoms()
        
        self.logfile = self.openfile(logfile, comm=world, mode='a')
        self.stress = stress
        self.peratom = peratom
        if self.dyn is not None:
            self.hdr = "%-9s %12s" % ("Time[ps]", "GFree[eV]")
            self.fmt = "%-10.4f %12.4f"
        else:
            self.hdr = ""
            self.fmt = ""
        if self.peratom:
            self.hdr += "%12s %12s %12s  %6s" % ("Etot/N[eV]", "Epot/N[eV]",
                                                 "Ekin/N[eV]", "T[K]")
            self.fmt += "%12.4f %12.4f %12.4f  %6.1f"
        else:
            self.hdr += "%12s %12s %12s  %6s" % ("Etot[eV]", "Epot[eV]",
                                                 "Ekin[eV]", "T[K]")
            # Choose a sensible number of decimals
            if global_natoms <= 100:
                digits = 4
            elif global_natoms <= 1000:
                digits = 3
            elif global_natoms <= 10000:
                digits = 2
            else:
                digits = 1
            self.fmt += 3 * ("%%12.%df " % (digits,)) + " %6.1f"
        if self.stress:
            self.hdr += ('      -----stress [GPa] --------     ----- BOX [A] ------ ')
            self.fmt += 3 * " %10.3f"
            self.fmt += 3 * " %8.3f"
        self.fmt += "\n"
        if header:
            self.logfile.write(self.hdr + "\n")

    def __del__(self):
        self.close()

    def __call__(self):
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = self.atoms.get_temperature()
        global_natoms = self.atoms.get_global_number_of_atoms()
        if self.peratom:
            epot /= global_natoms
            ekin /= global_natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000 * units.fs)
            gf = self.dyn.get_gibbs_free_energy()
            dat = (t,gf)
        else:
            dat = ()
        dat += (epot + ekin, epot, ekin, temp)
        if self.stress:
            stress = self.atoms.get_stress (include_ideal_gas=True)/units.GPa
            dat += (stress[0], stress[1], stress[2])
            #dat += tuple(self.atoms.get_stress(
            #    include_ideal_gas=True) / units.GPa)
            cell = self.atoms.cell.cellpar() 
            #cell = self.atoms.get_cell()
            dat += (cell[0], cell[1], cell[2])
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()
