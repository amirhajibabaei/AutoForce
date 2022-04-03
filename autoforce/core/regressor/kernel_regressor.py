# +
import autoforce.cfg as cfg
from ..dataclass import Conf, Target
from ..parameter import Cutoff, ChemPar
from ..function import Kernel
from ..descriptor import Descriptor
from .regressor import Regressor
from torch import Tensor
import torch
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Union, Any, Dict


class KernelRegressor(Regressor):
    """
    TODO:

    """

    def __init__(self,
                 descriptor: Descriptor,
                 kernel: Kernel,
                 exponent: ChemPar
                 ) -> None:
        """
        TODO:

        """
        self.descriptor = descriptor
        self.basis = descriptor.new_basis()
        self.kernel = kernel
        self.exponent = exponent

    @property
    def cutoff(self) -> Cutoff:
        return self.descriptor.cutoff

    def _kernel(self,
                s: int,
                uv: Tensor,
                u: Tensor,
                v: Tensor
                ) -> Tensor:
        return self.kernel.function(uv, u, v)**self.exponent[s]

    def get_design_matrix(self,
                          confs: Sequence[Conf],
                          ) -> (Tensor, Tensor, Tuple[Tuple[int, int], ...]):
        Ke = []
        Kf = []
        sections = tuple((s, c) for s, c in self.basis.count().items())
        for conf in confs:
            design_dict = self.get_design_dict(conf)
            ke = []
            kf = []
            for species, count in sections:
                if species in design_dict:
                    e, f = design_dict[species]
                else:
                    e = torch.zeros(1, count, dtype=cfg.float_t)
                    f = torch.zeros(conf.positions.numel(),
                                    count, dtype=cfg.float_t)
                ke.append(e)
                kf.append(f)
            Ke.append(torch.cat(ke, dim=1))
            Kf.append(torch.cat(kf, dim=1))
        Ke = torch.cat(Ke)
        Kf = torch.cat(Kf)
        return Ke, Kf, sections

    def get_design_dict(self,
                        conf: Conf,
                        ) -> Dict:
        design_dict = {}
        basis_norms = self.basis.norms()
        products, norms = self.descriptor.get_scalar_products_dict(conf,
                                                                   self.basis)
        for species in products.keys():
            kern = []
            kern_grad = []
            species_norms = torch.stack(norms[species]).view(1, -1)
            for a in zip(*products[species], basis_norms[species]):
                k = self._kernel(species,
                                 torch.stack(a[:-1]).view(1, -1),
                                 a[-1].view(1, 1),
                                 species_norms,
                                 ).sum()
                dk, = torch.autograd.grad(k,
                                          conf.positions,
                                          retain_graph=True)
                kern.append(k.detach())
                kern_grad.append(dk.view(-1, 1))
            kern = torch.stack(kern).view(1, -1)
            kern_grad = torch.cat(kern_grad, dim=1)
            design_dict[species] = (kern, -kern_grad)
        return design_dict

    def get_basis_overlaps(self) -> Dict:
        gram_dict = self.descriptor.get_gram_dict(self.basis)
        basis_norms = self.basis.norms()
        for species, gram in gram_dict.items():
            norms = torch.stack(basis_norms[species])
            gram_dict[species] = self._kernel(species,
                                              gram,
                                              norms.view(1, -1),
                                              norms.view(-1, 1))
        return gram_dict

    def set_weights(self,
                    weights: Tensor,
                    sections: Tuple[Tuple[int, int], ...]
                    ) -> None:
        """
        TODO:

        """
        species, count = zip(*sections)
        weights = torch.split(weights, count)
        self.weights = {s: w for s, w in zip(species, weights)}

    def get_target(self, conf: Conf) -> Target:
        """
        TODO:

        """
        basis_norms = self.basis.norms()
        products, norms = self.descriptor.get_scalar_products_dict(
            conf, self.basis)
        energy = 0
        for species, prod in products.items():
            k = self._kernel(species,
                             torch.stack([torch.stack(a) for a in prod]),
                             torch.stack(norms[species]).view(-1, 1),
                             torch.stack(basis_norms[species]).view(1, -1),
                             )
            energy = energy + (k @ self.weights[species]).sum()
        if energy.grad_fn:
            g, = torch.autograd.grad(energy, conf.positions, retain_graph=True)
        else:
            g = 0
        return Target(energy=energy.detach(), forces=-g)
