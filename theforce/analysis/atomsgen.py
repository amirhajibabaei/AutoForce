# +
import datetime
import itertools
import os

import numpy as np

from theforce.analysis.simplesim import SimpleSim


def date(fmt="%m/%d/%Y %H:%M:%S"):
    return datetime.datetime.now().strftime(fmt)


def top(x, y, depth):
    arg = np.argsort(y)
    s = min(depth, len(y))
    xx = [x[k] for k in arg[:s]]
    yy = [y[k] for k in arg[:s]]
    return xx, yy


def rand(x, m):
    n = len(x)
    if n > m:
        cap = min(n, m)
        y = np.random.permutation(n)[:cap]
        return [x[k] for k in y]
    else:
        return x


def abstract(generator):
    """
    switch pairs that cancel each other are eliminated.
    """
    _abs = []
    for switch in generator:
        index, i, f = switch
        if i != f:
            reverse = (index, f, i)
            if reverse in _abs:
                _abs.remove(reverse)
            else:
                _abs.append(switch)
    return tuple(_abs) == generator


def heuristic(parent, switch):
    """
    rules: ?
    """
    k, i, f = switch
    if (k, f, i) in parent:
        return False
    for kk, ii, ff in parent:
        if kk > k and ii == i and ff == f:
            return False
    return True


def canonical(generator):
    """
    path independent, unique, and abstract representation of a generator.
    """
    status = {}
    for k, i, f in generator:
        if k in status:
            assert status[k][1] == i
            status[k] = (status[k][0], f)
        else:
            status[k] = (i, f)
    return tuple((k, *status[k]) for k in sorted(status.keys()))


class AtomsGenerator:
    """
    terms:

    switch: a switch is defined by (index, i, f) which changes the atomic number
            of "index" from "i" to "f".

    switch_type: a switch without index, e.g. (i, f)

    generator: a generator is a tuple sequence of switches.

    """

    def __init__(self, atoms, sim=1.0 - 1e-6, forbidden={}, prefix="search", rank=0):
        """
        atoms: an ASE Atoms object with an attached calculator
        sim: site similarity threshold
        forbidden: {Z: [i1, i2, ...]}, Z is not allowed at given indices
        prefix: starting string for output files names
        rank: for parallel io, only rank 0 will do the writing

        self.dry_run: if True, energy calculations will be skipped.

        """
        self.atoms = atoms
        self.sim = sim
        self.simkern = SimpleSim(atoms)
        self.forbidden = forbidden
        self.prefix = prefix
        self.rank = rank
        self.log("hello atoms generator", "w")
        self.cached = {}
        self.cachfile = f"{self.prefix}.cached"
        self.read_cached()

        self.dry_run = False

    def log(self, msg, mode="a"):
        if self.rank == 0:
            with open(f"{self.prefix}.log", mode) as f:
                f.write(f"{date()} {msg}\n")

    def read_cached(self):
        if os.path.isfile(self.cachfile):
            with open(self.cachfile) as cache:
                lines = cache.readlines()
                for line in lines:
                    x, y = line.split(":")
                    self.cached[eval(x)] = float(y)
                self.log(f"{len(lines)} energies are read from {self.cachfile}")

    def get_potential_energy(self, generator):
        try:
            e = self.cached[generator]
        except KeyError:
            if self.dry_run:
                e = 0.0
            else:
                self.do(generator)
                e = self.atoms.get_potential_energy()
                self.undo(generator)
            self.cached[generator] = e
            if self.rank == 0:
                with open(self.cachfile, "a") as cache:
                    cache.write(f"{generator} : {e}\n")
        return e

    def get_potential_energies(self, generators):
        energies = []
        for generator in generators:
            energies.append(self.get_potential_energy(generator))
        return energies

    def do(self, generator):
        for index, i, f in generator:
            assert self.atoms[index].number == i
            self.atoms[index].number = f

    def undo(self, generator):
        for index, i, f in generator[::-1]:
            assert self.atoms[index].number == f
            self.atoms[index].number = i

    def generate(self, parents, switch_type):
        generation = set()
        for parent in parents:
            self.do(parent)
            i, f = switch_type
            unique = []
            for at in self.atoms:
                if at.number == i:
                    if f in self.forbidden and at.index in self.forbidden[f]:
                        continue
                    is_unique = True
                    for u in unique:
                        if self.simkern(u, at.index, threshold=self.sim):
                            is_unique = False
                            break
                    if not is_unique:
                        continue
                    unique += [at.index]
                    switch = (at.index, i, f)
                    if heuristic(parent, switch):
                        generation.add(canonical((*parent, switch)))
            self.undo(parent)
        return generation

    def swaps(self, parents, switch_type):
        a = self.generate(self.generate(parents, switch_type), switch_type[::-1])
        b = self.generate(self.generate(parents, switch_type[::-1]), switch_type)
        return a.union(b)

    def search_swaps(self, parents, swap_types, epochs=1, max_child=10, max_parents=10):
        for _ in range(epochs):
            generation = set()
            for parent in parents:
                children = set()
                for st in swap_types:
                    children = children.union(self.swaps([parent], st))
                generation = generation.union(set(rand(list(children), max_child)))
                generation.add(parent)
            generation = list(generation)
            energies = self.get_potential_energies(generation)
            parents, energies = top(generation, energies, max_parents)
            self.log("")
            self.log("")
            self.log("")
            self.log(f"{len(parents)} lowest energies of {len(generation)}")
            for p, e in zip(*[parents, energies]):
                self.log(f"{e} {p}")
        return parents

    def savegen(self, generation, file):
        with open(file, "w") as f:
            for g in generation:
                f.write(f"{g}\n")

    def loadgen(self, file):
        gen = []
        with open(file, "r") as f:
            for g in f.readlines():
                gen.append(eval(g.strip()))
        return gen
