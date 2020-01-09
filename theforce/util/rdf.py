#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import pi
import torch
import numpy as np
import itertools


def get_numbers_pairs(atoms_numbers, numbers, pairs):
    if pairs:
        numbers = sorted(list(set([a for b in pairs for a in b])))
    elif numbers:
        pairs = ([(a, b) for a, b in itertools.combinations(numbers, 2)] +
                 [(a, a) for a in numbers])
    else:
        numbers = np.unique(atoms_numbers)
        pairs = ([(a, b) for a, b in itertools.combinations(numbers, 2)] +
                 [(a, a) for a in numbers])
    print(f'numbers: {numbers} \npairs: {pairs}')
    return numbers, pairs


def rdf(data, rmax, bins=100, rmin=0., numbers=None, pairs=None):

    # numbers, pairs
    numbers, pairs = get_numbers_pairs(data[0].numbers, numbers, pairs)

    # hists
    binargs = dict(bins=bins, min=rmin, max=rmax)
    density = {number: 0 for number in numbers}
    hist = {pair: torch.histc(torch.empty(0), **binargs) for pair in pairs}
    count = {pair: 0 for pair in pairs}

    snaps = 0
    for atoms in data:
        snaps += 1

        # densities
        nums, freq = atoms.tnumbers.unique(return_counts=True)
        for n, f in zip(*[nums.tolist(), freq.tolist()]):
            try:
                density[n] += f/atoms.get_volume()
            except:
                pass

        # distances
        atoms.update(cutoff=rmax)
        for pair in pairs:
            for loc, atomic in zip(*[atoms, atoms.numbers]):
                if atomic != pair[0]:
                    continue
                loc.select(*pair, bothways=True)
                count[pair] += 1
                hist[pair] += torch.histc(loc.r.pow(2).sum(dim=-1).sqrt(),
                                          **binargs)
    for number in numbers:
        density[number] /= snaps

    r = torch.linspace(rmin, rmax, bins)
    dr = r[1]-r[0]
    g = {pair: hist[pair]/(count[pair]*4*pi*r**2*dr*density[pair[1]])
         for pair in pairs}

    return r, g


def _rdf(data, rmax, bins=100, rmin=0., numbers=None, pairs=None):
    """
    This uses all distances rather than neighborlist which may be faster
    if the number of atoms is not too large but it may fail for a small 
    box because it doesn't consider images due to pbc.
    """

    # numbers, pairs
    numbers, pairs = get_numbers_pairs(data[0].numbers, numbers, pairs)

    # hists
    binargs = dict(bins=bins, min=rmin, max=rmax)
    density = {number: 0 for number in numbers}
    hist = {pair: torch.histc(torch.empty(0), **binargs) for pair in pairs}
    count = {pair: 0 for pair in pairs}

    snaps = 0
    for atoms in data:
        snaps += 1

        # densities
        nums, freq = np.unique(atoms.numbers, return_counts=True)
        for n, f in zip(*[nums.tolist(), freq.tolist()]):
            try:
                density[n] += f/atoms.get_volume()
            except:
                pass

        # distances
        dij = torch.from_numpy(atoms.get_all_distances(mic=True))
        for pair in pairs:
            i = torch.from_numpy(atoms.numbers == pair[0])
            j = torch.from_numpy(atoms.numbers == pair[1])
            r = dij[i][:, j]
            count[pair] += r.shape[0]
            hist[pair] += torch.histc(r.view(-1),
                                      **binargs)
    for number in numbers:
        density[number] /= snaps

    r = torch.linspace(rmin, rmax, bins)
    dr = r[1]-r[0]
    g = {pair: hist[pair]/(count[pair]*4*pi*r**2*dr*density[pair[1]])
         for pair in pairs}

    return r, g


# TODO: write an rdf using nl

