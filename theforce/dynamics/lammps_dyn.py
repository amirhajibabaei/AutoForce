# +
"""
Defines ASE-type LammpsDynamics class:
    dyn = LammpsDynamics(atoms, 2.*units.fs, 'md.traj')
    dyn.run(commands_string)
    
Notes:
    It converts quantities to 'metal' units before passing them to LAMMPS.
    In commands_string, quantities should be given in the LAMMPS 'metal' units.

References:
    Virial conversion constant nktv2p:
        https://github.com/lammps/lammps/blob/master/src/update.cpp
    Example:
        https://github.com/lammps/lammps/blob/master/examples/COUPLE/fortran_dftb/simple.f90
"""
from lammps import lammps
import numpy as np
from ase.md.md import MolecularDynamics
from ase.calculators.lammps import convert


class LammpsDynamics(MolecularDynamics):

    def __init__(self, atoms, timestep, trajectory=None, logfile=None,
                 loginterval=1, append_trajectory=False):
        super().__init__(atoms, timestep, trajectory, logfile,
                         loginterval, append_trajectory=append_trajectory)
        self.lmp = make_lammps(self.atoms)
        dt = convert(self.dt, 'time', 'ASE', 'metal')
        self.lmp.command(f'timestep {dt}')

        def callback(caller, ntimestep, nlocal, tag, pos, fext):

            sync_atoms(self.atoms, self.lmp, pos)
            f = self.atoms.get_forces()
            fext[:] = convert(f, 'force', 'ASE', 'metal')

            if True:  # pass energy/stress to LAMMPS
                energy = self.atoms.get_potential_energy()
                energy = convert(energy, 'energy', 'ASE', 'metal')
                self.lmp.fix_external_set_energy_global('ext', energy)
                if 'stress' in self.atoms.calc.implemented_properties:
                    vir = self.atoms.get_stress()
                    vir = convert(vir, 'pressure', 'ASE', 'metal')
                    volume = atoms.get_volume()
                    nktv2p = 1.6021765e6  # for metal units
                    vir = -vir / (nktv2p/volume)
                    vir[3:] = vir[3:][::-1]
                    self.lmp.fix_external_set_virial_global('ext', vir)

            # probably writing to traj in the middle of timestepping!
            self.call_observers()
            # TODO: bug?
            #T1 = self.lmp.get_thermo('temp')
            #T2 = self.atoms.get_temperature()
            # print(T1/T2) # prints ~1.01

        # is this needed?
        #self.lmp.command("pair_style zero 0.1")
        #self.lmp.command("pair_coeff * *")

        #
        self.lmp.command("fix ext all external pf/callback 1 1")
        self.lmp.set_fix_external_callback("ext", callback)
        self.lmp.command("fix_modify ext energy yes virial yes")

    def run(self, commands_string):
        self.lmp.commands_string(commands_string)


def make_lammps(atoms):

    # coordinates, velocities
    natoms = atoms.get_global_number_of_atoms()
    cell = convert(atoms.cell, 'distance', 'ASE', 'metal')
    xhi, xy, xz, _yx, yhi, yz, _zx, _zy, zhi = cell.flatten()
    assert np.allclose([_yx, _zx, _zy], 0.)
    pos = convert(atoms.positions, 'distance', 'ASE', 'metal').ravel()
    vel = convert(atoms.get_velocities(), 'velocity', 'ASE', 'metal').ravel()
    pbc = list(map({True: 'p', False: 'f'}.get, atoms.pbc))

    # types
    un, ui = np.unique(atoms.numbers, return_index=True)
    nt = len(ui)
    ut = np.arange(1, nt+1, dtype=int)
    typemap = {a: b for a, b in zip(un, ut)}
    types = np.array(list(map(typemap.get, atoms.numbers)))
    print(f"ASE-LAMMPS type map: {types}")

    # commands
    cmds = ["units metal",
            "atom_style atomic",
            "atom_modify map array sort 0 0",
            "boundary {} {} {}".format(*pbc),
            f"region cell prism  0 {xhi} 0 {yhi} 0 {zhi} {xy} {xz} {yz} units box",
            f"create_box {nt} cell"
            ]

    lmp = lammps()
    lmp.commands_list(cmds)
    ids = [a.index+1 for a in atoms]  # indices start from 1?
    lmp.create_atoms(natoms, ids, types, pos, v=vel.tolist())

    # masses
    um = convert(atoms.get_masses()[ui], 'mass', 'ASE', 'metal')
    for t, m in zip(ut, um):
        lmp.command(f"mass {t} {m}")

    return lmp


def sync_atoms(atoms, lmp, lmp_pos=None):
    boxlo, [xhi, yhi, zhi], xy, yz, xz, periodicity, box_change = lmp.extract_box()
    cell = np.array([[xhi, xy, xz],
                     [0., yhi, yz],
                     [0., 0., zhi]])
    atoms.cell = convert(cell, 'distance', 'metal', 'ASE')
    if lmp_pos is None:
        pos = lmp.numpy.extract_atom('x')
    else:
        pos = lmp_pos
    atoms.positions = convert(pos, 'distance', 'metal', 'ASE')
    vel = convert(lmp.numpy.extract_atom('v'), 'velocity', 'metal', 'ASE')
    atoms.set_velocities(vel)
    # TODO: check if these velocities correct!
    # At what point during timestepping?
    # check possible connection with the small error in temperatures.


def example():
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase import units
    from theforce.util.aseutil import init_velocities

    # create atoms
    atoms = bulk('Au', cubic=True).repeat(3*[3])
    atoms.rattle(0.1)
    atoms.calc = EMT()

    # define and run dynamics
    temp = 1000.
    press = convert(1.7*units.GPa, 'pressure', 'ASE', 'metal')
    commands_string = f"""
    fix 1 all npt temp {temp} {temp} $(100.0*dt) iso {press} {press} $(1000.0*dt)
    thermo_style custom step temp press pe ke etotal vol enthalpy pxx pyy pzz pxy pxz pyz
    thermo 1
    dump whatever all custom 1 dump.atom id type x y z vx vy vz fx fy fz
    run 100
    """

    init_velocities(atoms, temp)
    dyn = LammpsDynamics(atoms, 2.*units.fs, 'md.traj')
    dyn.run(commands_string)


if __name__ == '__main__':
    example()
