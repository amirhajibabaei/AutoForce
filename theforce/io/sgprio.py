# +
from theforce.descriptor.atoms import Local
from theforce.util.parallel import rank
from ase.io import read
from ase.atoms import Atoms
import torch
import numpy as np
import io


def write_lce(loc, f):
    lines = [f'{loc.number:4d}\n']
    for s, r in zip(loc._b, loc._r):
        lines.append('{:4d} {:16.8f} {:16.8f} {:16.8f}\n'.format(
            int(s), *r.tolist()))
    for line in lines:
        f.write(line)
    return lines


def read_lce(blk):
    a = int(blk[0].strip())
    b = []
    r = []
    for line in blk[1:]:
        s = line.split()
        b.append(s[0])
        r.append(s[1:])
    b = np.array(b, dtype=int)
    r = np.array(r, dtype=float)
    r = torch.from_numpy(r)
    i = 0
    j = np.array(range(1, b.shape[0]+1))
    obj = Local(i, j, a, b, r)
    return obj


def convert_block(typ, blk):
    if typ == 'atoms':
        obj = read(io.StringIO(''.join(blk)), format='xyz')
    elif typ == 'local':
        obj = read_lce(blk)
    else:
        raise RuntimeError(f'type {typ} is unknown')
    return obj


class SgprIO:

    def __init__(self, path):
        self.path = path

    def write(self, obj):
        if type(obj) == Local:
            self.write_lce(obj)
        elif type(obj) == Atoms:
            self.write_atoms(obj)
        else:
            raise RuntimeError(f'no recipe for {type(obj)}')

    def write_lce(self, loc):
        if rank() == 0:
            with open(self.path, 'a') as f:
                f.write('\nstart: local\n')
                write_lce(loc, f)
                f.write('end: local\n')

    def write_atoms(self, atoms):
        if rank() == 0:
            with open(self.path, 'a') as f:
                f.write('\nstart: atoms\n')
            atoms.write(self.path, format='xyz', append=True)
            with open(self.path, 'a') as f:
                f.write('end: atoms\n')

    def read(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()
        on = False
        data = []
        for line in lines:
            if line.startswith('start:'):
                on = True
                typ = line.split()[-1]
                blk = []
            elif line.startswith('end:'):
                assert line.split()[-1] == typ
                on = False
                obj = convert_block(typ, blk)
                data.append((typ, obj))
            elif on:
                blk.append(line)
        return data
