# +
from .des import LocalDes
from collections import defaultdict
import itertools
from typing import Dict


class Basis:

    __slots__ = ('index',
                 'descriptors',
                 'active')

    def __init__(self):
        self.index = None
        self.descriptors = defaultdict(list)
        self.active = defaultdict(list)

    def append(self, d: LocalDes) -> None:
        self.descriptors[d.species].append(d.detach())
        self.active[d.species].append(True)

    def count(self) -> Dict:
        return {s: a.count(True) for s, a in self.active.items()}

    def norms(self) -> Dict:
        return {s: [d.norm for d in itertools.compress(self.descriptors[s], a)]
                for s, a in self.active.items()}
