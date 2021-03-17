# +
import os
from shutil import which
from io import StringIO
import re
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
import numpy as np


class GaussianCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, command=None):
        """
        command: 'path_to_gxx < input > output'
        gxx: g16, g09, or g03
        """
        Calculator.__init__(self)
        if command:
            self.args = parse_command(command)
        else:
            self.args = (get_gex(), 'Gaussian.com', 'Gaussian.log')
        self.blocks = get_blocks(self.args[1])

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        tmp = StringIO()
        self.atoms.write(tmp, format='gaussian-in')
        blocks = get_blocks(tmp)
        self.blocks[2] = blocks[2]
        write_blocks(self.blocks, file=self.args[1])
        assert os.system('{} < {} > {}'.format(*self.args)) == 0
        output = read(self.args[2], format='gaussian-out')
        self.calc = output.calc
        self.results = output.calc.results
        self.results['stress'] = np.zeros(6)


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
