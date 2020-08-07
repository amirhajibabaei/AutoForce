# +
from theforce.util.simplesim import SimpleSim
import datetime
import itertools
import numpy as np
import os


class BruteDoping:
    """
    A deterministic algorithm for finding minimum energy doping.
    A doping sequence is defined by: 
       ((i1, x1, y1), (i2, x2, y2), ...)
    where i1 is the index of the atom whose atomic number has 
    changed from x1 to y1, etc.
    """

    def __init__(self, atoms, prefix='doping', callback=None, rank=0):
        """
        atoms:    an ASE Atoms object with an attached calculator
        prefix:   starting string for output files names
        callback: a function that is called after each generation 
                  e.g. callback=(func, arg)
        rank:     for parallel io, only rank 0 will do the writing

        self.dry_run = True can be set which skips energy calculations,
        for testing purposes.

        """

        self.atoms = atoms
        self.prefix = prefix
        self.callback = callback
        self.rank = rank
        self.log('brute doping', 'w')
        self.cached = {}
        self.cachfile = f'{self.prefix}.cached'
        self.read_cached()
        self.sim = SimpleSim(atoms)
        self.dry_run = False

    def log(self, msg, mode='a'):
        if self.rank == 0:
            with open(f'{self.prefix}.log', mode) as f:
                f.write(f'{date()} {msg}\n')

    def read_cached(self):
        if os.path.isfile(self.cachfile):
            with open(self.cachfile) as cache:
                lines = cache.readlines()
                for line in lines:
                    x, y = line.split(':')
                    self.cached[eval(x)] = float(y)
                self.log(
                    f'{len(lines)} energies are read from {self.cachfile}')

    def get_potential_energy(self, new):
        try:
            e = self.cached[frozenset(new)]
        except KeyError:
            if self.dry_run:
                e = 0.
            else:
                e = self.atoms.get_potential_energy()
            self.cached[frozenset(new)] = e
            if self.rank == 0:
                with open(self.cachfile, 'a') as cache:
                    cache.write(f'{frozenset(new)} : {e}\n')
        return e

    def dope(self, dopings):
        for index, i, f in dopings:
            assert self.atoms[index].number == i
            self.atoms[index].number = f

    def undope(self, dopings):
        for index, i, f in dopings:
            assert self.atoms[index].number == f
            self.atoms[index].number = i

    def search(self, deltas, depth=1, similar=0.999, forbidden={}):
        """
        deltas: e.g. {3: -1, 11: 1} replaces 1 Li with Na
        depth: keeps only "depth" lowest energy children of each parent.
        similar: 0-to-1, used for skipping similar sites. 
        If similar=1 -> all atoms are unique
        forbidden: {species: [indices]}, forbids certain species from 
        placement at certain sites.

        if depth is None, all children will be kept.

        TODO: find unique sites for speedup

        """
        self.log(f'searching depth: {depth}')
        if depth is None:
            self.log('None -> all dopings will be generated (one path is enough)')
        self.log(f'forbidden: {forbidden}')
        global_ = []
        global_e = []
        for path in doping_paths(deltas):
            self.log(f'path: {path}')
            parents = [tuple()]
            for r, a in path:
                generation = []
                energies = []
                skipped = 0
                for dopings in parents:
                    self.dope(dopings)  # 1
                    childs = []
                    ch_energies = []
                    unique = []
                    for at in self.atoms:
                        if at.number == r:
                            if a in forbidden and at.index in forbidden[a]:
                                continue
                            is_unique = True
                            for u in unique:
                                if self.sim(u, at.index, threshold=similar):
                                    is_unique = False
                                    break
                            if not is_unique:
                                skipped += 1
                                continue
                            unique += [at.index]
                            head = (at.index, r, a)
                            self.dope((head,))  # 2
                            new = (*dopings, head)
                            e = self.get_potential_energy(new)
                            ch_energies += [e]
                            childs += [new]
                            self.undope((head,))  # 2
                    self.undope(dopings)  # 1
                    childs, ch_energies = top(childs, ch_energies, depth)
                    generation += childs
                    energies += ch_energies
                self.log(f'generation size: {len(generation)} skipped :{skipped}')
                parents = generation
                if self.callback:
                    self.callback[0](self.callback[1])
            generation, energies = top(generation, energies, 5)
            self.log('top contenders:')
            for x, y in zip(*[generation, energies]):
                self.log(f'{x} {y}')
            global_ += generation
            global_e += energies
            if depth is None:
                break
        self.log('search ended with global minimum:')
        global_, global_e = top(global_, global_e, 1)
        x = global_[0]
        e = global_e[0]
        self.log(f'{x} {e}')
        self.dope(x)
        e = self.get_potential_energy(x)
        if self.rank == 0:
            self.atoms.write(f'{self.prefix}.cif')
        minimum = self.atoms.copy()
        self.undope(x)
        return minimum


def date(fmt="%m/%d/%Y %H:%M:%S"):
    return datetime.datetime.now().strftime(fmt)


def doping_paths(delta):
    add = []
    remove = []
    for x, y in delta.items():
        if y > 0:
            add += y*[x]
        elif y < 0:
            remove += (-y)*[x]
    additions = set()
    for a in itertools.permutations(add):
        additions.add(a)
    removals = set()
    for r in itertools.permutations(remove):
        removals.add(r)
    paths = set()
    for x in itertools.product(removals, additions):
        paths.add(tuple((r, a) for r, a in zip(*[*x])))
    return paths


def top(x, y, depth):
    arg = np.argsort(y)
    s = min(depth, len(y))
    xx = [x[k] for k in arg[:s]]
    yy = [y[k] for k in arg[:s]]
    return xx, yy
