# +
import autoforce.cfg as cfg
import autoforce.core as core
from torch import Tensor
import torch
from typing import Sequence, Tuple


class Shift(core.Regressor):
    """
    TODO:

    """

    @property
    def cutoff(self) -> None:
        return None

    def set_weights(self,
                    weights: Tensor,
                    sections: Tuple[int]
                    ) -> None:
        self.weights = {s: a for s, a in zip(sections, weights)}

    def get_design_matrix(self,
                          confs: Sequence[core.Conf]
                          ) -> (Tensor, Tensor, Tuple[int]):
        sections = set()
        for conf in confs:
            sections.update(conf.unique_counts.keys())
        sections = tuple(sections)
        dim = len(sections)
        index = {s: i for i, s in enumerate(sections)}
        e = []
        f_len = 0
        for conf in confs:
            v = dim*[0]
            for z, c in conf.unique_counts.items():
                v[index[z]] = c
            e.append(v)
            f_len += conf.number_of_atoms
        e = torch.tensor(e, dtype=cfg.float_t)
        f = torch.zeros(3*f_len, dim, dtype=cfg.float_t)
        return e, f, sections

    def get_target(self, conf):
        e = 0
        for number, count in conf.unique_counts.items():
            e = e + self.weights[number]*count
        return core.Target(energy=e, forces=0)
