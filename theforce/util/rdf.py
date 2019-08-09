#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import pi
import torch


def rdf(data, min=1., max=5., bins=100):
    binargs = dict(bins=bins, min=min, max=max)
    numbers = data.numbers_set()
    pairs = data.pairs_set(numbers)
    density = {number: 0 for number in numbers}
    hist = {pair: torch.histc(torch.empty(0), **binargs) for pair in pairs}
    count = {pair: 0 for pair in pairs}

    snaps = 0
    for atoms in data.X[-1000::10]:
        snaps += 1

        # densities
        nums, freq = atoms.tnumbers.unique(return_counts=True)
        for n, f in zip(*[nums.tolist(), freq.tolist()]):
            density[n] += f/atoms.get_volume()

        # distances
        atoms.update(cutoff=max)
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

    r = torch.linspace(min, max, bins)
    dr = r[1]-r[0]
    g = {pair: hist[pair]/(count[pair]*4*pi*r**2*dr*density[pair[1]])
         for pair in pairs}

    return r, g

