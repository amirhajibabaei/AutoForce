# +
from ase.calculators.vasp import Vasp2
import re
import os


def get_command():
    if os.path.isfile('COMMAND'):
        c = ''.join(open('COMMAND').readlines()).replace('\n', ' ')
        command = re.sub(' +', ' ', c)
    else:
        if 'CORES_FOR_VASP' in os.environ:
            cores_for_vasp = os.environ['CORES_FOR_VASP']
        elif 'CORES_FOR_ML' in os.environ:
            cores_for_vasp = os.cpu_count() - int(os.environ['CORES_FOR_ML'])
        else:
            cores_for_vasp = os.cpu_count()
        command = f'mpirun -n {cores_for_vasp} vasp_std'
        with open('_vasp_command', 'w') as f:
            f.write(command)
    return command


def get_setups():
    setups = {}
    if os.path.isfile('SETUPS'):
        for _line in open('SETUPS'):
            if '#' in _line:
                line = _line[:_line.index('#')]
            else:
                line = _line
            if len(line.split()) == 0:
                continue
            if '=' in line:
                a, b = line.split('=')
            else:
                a, b = line.split()
            setups[a.strip()] = b.strip()
    return setups


def preprocess_atoms(atoms):
    if os.path.isfile('IMAG'):
        imag = {}
        for _line in open('IMAG'):
            if '#' in _line:
                line = _line[:_line.index('#')]
            else:
                line = _line
            if len(line.split()) == 0:
                continue
            if '=' in line:
                a, b = line.split('=')
            else:
                a, b = line.split()
            imag[int(a)] = float(b)
        m = [imag[z] if z in imag else 0. for z in atoms.numbers]
        atoms.set_initial_magnetic_moments(m)


command = get_command()
calc = Vasp2(command=command,
             setups=get_setups(),
             directory='vasp')
if os.path.isfile('INCAR'):
    calc.read_incar()
if os.path.isfile('KPOINTS'):
    calc.read_kpoints()
