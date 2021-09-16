# +
from lammps import lammps
from ase.atoms import Atoms
from ase.calculators.lammps import convert
import numpy as np


def read_lammps_file(file):
    commands = []
    units = None
    fixID = None
    run = None
    scope = {}
    for line in open(file):
        if line.lower().startswith('#autoforce'):
            exec(line[10:].strip(), scope)
            continue
        if '#' in line:
            line = line[:line.index('#')]
        line = ' '.join(line.split())
        if line == '':
            continue
        if line.startswith('units'):
            units = line.split()[1]
        if line.lower().startswith('fix autoforce'):
            fixID = line.split()[1]
        if line.startswith('run'):
            run = line
        else:
            commands.append(line)
    map_numbers = scope['atomic_numbers']
    return units, map_numbers, fixID, commands, run


def get_cell():
    global lmp
    boxlo, [xhi, yhi, zhi], xy, yz, xz, pbc, box_change = lmp.extract_box()
    cell = np.array([[xhi, xy, xz],
                     [0., yhi, yz],
                     [0., 0., zhi]])
    return cell, pbc


def callback(caller, ntimestep, nlocal, tag, pos, fext):
    global lmp, atoms, units, map_numbers, fixID, nktv2p, calc

    # build atoms
    cell, pbc = get_cell()
    cell = convert(cell, 'distance', units, 'ASE')
    positions = convert(pos, 'distance', units, 'ASE')
    if atoms is None:
        numbers = lmp.numpy.extract_atom('type')
        numbers = list(map(map_numbers.get, numbers))
        atoms = Atoms(numbers=numbers, positions=positions, pbc=pbc, cell=cell)
        atoms.calc = calc
    else:
        atoms.cell = cell
        atoms.positions = positions

    # calculate energy, force, and virial
    f = atoms.get_forces()
    e = atoms.get_potential_energy()
    fext[:] = convert(f, 'force', 'ASE', units)
    e = convert(energy, 'energy', 'ASE', units)
    lmp.fix_external_set_energy_global(fixID, e)
    if 'stress' in atoms.calc.implemented_properties:
        v = atoms.get_stress()
        v = convert(vir, 'pressure', 'ASE', units)
        vol = atoms.get_volume()
        v = -v / (nktv2p[units]/vol)
        v[3:] = v[3:][::-1]
        lmp.fix_external_set_virial_global(fixID, v)


nktv2p = {"lj": 1.0,
          "real": 68568.415,
          "metal": 1.6021765e6,
          "si": 1.0,
          "cgs": 1.0,
          "electron": 2.94210108e13,
          "micro": 1.0,
          "nano": 1.0,
          }

if __name__ == '__main__':
    # command line args:
    import argparse
    parser = argparse.ArgumentParser(
        description='Dynamics with LAMMPS')
    parser.add_argument('-i', '--input',
                        default='in.lammps', type=str,
                        help='LAMMPS input script')
    args = parser.parse_args()

    # main:
    import theforce.cl as cline
    calc = cline.gen_active_calc()
    atoms = None
    units, map_numbers, fixID, commands, run = read_lammps_file(args.input)
    lmp = lammps()
    lmp.commands_list(commands)
    lmp.set_fix_external_callback(fixID, callback)
    lmp.command(run)
