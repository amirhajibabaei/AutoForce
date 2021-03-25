# +
import os
from shutil import which
from io import StringIO
import re
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from theforce.util.util import mkdir_p
import numpy as np


class GaussianCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, command=None, wd='gaussian_wd', subtract=False):
        """
        command: 'path_to_gxx < input > output'
        gxx: g16, g09, or g03
        """
        Calculator.__init__(self)
        if command:
            self.args = parse_command(command)
        else:
            self.args = (get_gex(), 'Gaussian.gjf', 'Gaussian.log')
        self.blocks = get_blocks(self.args[1])
        self.charge_spin = self.blocks[2][0]
        self.wd = wd
        self.subtract = subtract
        self._single_atom_energy = {}

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # calculate energy-subtract
        subtract = 0
        if self.subtract:
            for a in self.atoms:
                subtract += self.single_atom_energy(a.symbol)
        # single-point calculation
        output = self._run(self.atoms)
        if output is None:
            raise RuntimeError('gaussian calculation failed!')
        self.calc = output.calc
        self.results = output.calc.results
        # set dummy stress
        if 'stress' not in self.results:
            self.results['stress'] = np.zeros(6)
        # subtract single atom energies
        self.results['energy'] -= subtract

    def single_atom_energy(self, symbol):
        if symbol not in self._single_atom_energy:
            file = f'subtract_energy_per_{symbol}'
            if os.path.isfile(f'set_{file}'):
                with open(f'set_{file}', 'r') as f:
                    energy = float(f.read())
            else:
                atoms = Atoms(symbol)
                output = self._run(atoms)
                if output is None:
                    energy = 0.  # here, do not raise an error
                else:
                    energy = output.get_potential_energy()
                with open(file, 'w') as f:
                    f.write(f'{energy}\n')
            self._single_atom_energy[symbol] = energy
        return self._single_atom_energy[symbol]

    def _run(self, atoms):
        """
        it returns output (atoms) with single-point-calc
        if the system call failes, it returns None
        """
        tmp = StringIO()
        atoms.write(tmp, format='gaussian-in')
        blocks = get_blocks(tmp)
        self.blocks[2] = blocks[2]
        self.blocks[2][0] = self.charge_spin
        cwd = os.getcwd()
        mkdir_p(self.wd)
        os.chdir(self.wd)
        os.system('rm -f *')
        write_blocks(self.blocks, file=f'{self.args[1]}')
        ierr = os.system('{} < {} > {}'.format(*self.args))
        if ierr == 0:
            output = read(self.args[2], format='gaussian-out')
        else:
            output = None
        os.chdir(cwd)
        return output


def get_gex():
    gex = None
    for gau in ('g16', 'g09', 'g03'):
        if which(gau):
            gex = gau
            break
    if gex is None:
        raise RuntimeError('no Gaussian executable found!')
    return gex


def parse_command(command):
    i = command.index('<')
    j = command.index('>')
    gex = command[:i].strip()
    gin = command[i+1:j].strip()
    gout = command[j+1:].strip()
    return gex, gin, gout


def get_blocks(obj):
    """
    blocks are things separated by a blank line.
    A Gaussian input file has 3 blocks.
    """
    lines = get_lines(obj)
    index = 0
    blocks = [[], [], []]
    for line in lines:
        if line == '':
            if index < 2:
                index += 1
        else:
            blocks[index].append(line)
    return blocks


def get_lines(obj):
    """
    no '\n' at the end of lines.
    """
    if type(obj) == StringIO:
        lines = obj.getvalue().split('\n')
    elif type(obj) == str:
        with open(obj, 'r') as f:
            lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


def write_blocks(blocks, file):
    with open(file, 'w') as f:
        for block in blocks:
            for line in block:
                f.write(line+'\n')
            f.write('\n')


def get_command():
    command = None
    if os.path.isfile('COMMAND'):
        c = ''.join(open('COMMAND').readlines()).replace('\n', ' ')
        command = re.sub(' +', ' ', c)
    return command


calc = GaussianCalculator(command=get_command())
