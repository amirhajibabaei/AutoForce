# +
import autoforce.cfg as cfg
from autoforce.core.dataclasses import Conf
from autoforce.core.descriptor import Descriptor
import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class Kernel(ABC):

    @abstractmethod
    def forward(self,
                uv: Tensor,
                u: Tensor,
                v: Tensor
                ) -> Tensor:
        """
        uv:   scalar products matrix <u_i,v_j> with shape (m, n)
        u:    norms sqrt(<u_i,u_i>) with shape (m, 1)
        v:    norms sqrt(<v_j,v_j>) with shape (1, n)

        output:
              a tensor with the same shape as uv
        """
        ...

    def get_design_matrix(self,
                          descriptor: Descriptor,
                          confslist: List[Conf]
                          ) -> (Tuple, Tensor, Tensor):
        Ke = []
        Kf = []
        basis_count = tuple((s, c) for s, c in
                            descriptor.basis_count().items())
        for conf in confslist:
            species_matrix = self.get_species_design_matrix(descriptor, conf)
            ke = []
            kf = []
            for species, count in basis_count:
                if species in species_matrix:
                    e, f = species_matrix[species]
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
        return basis_count, Ke, Kf

    def get_species_design_matrix(self,
                                  descriptor: Descriptor,
                                  conf: Conf
                                  ) -> Dict:
        species_matrix = {}
        basis_norms = descriptor.basis_norms()
        products, norms = descriptor.get_scalar_products(conf)
        for species in products.keys():
            kern = []
            kern_grad = []
            species_norms = torch.stack(norms[species]).view(1, -1)
            for a in zip(*products[species], basis_norms[species]):
                k = self.forward(torch.stack(a[:-1]).view(1, -1),
                                 a[-1].view(1, 1),
                                 species_norms
                                 ).sum()
                dk, = torch.autograd.grad(k,
                                          conf.positions,
                                          retain_graph=True)
                kern.append(k.detach())
                kern_grad.append(dk.view(-1, 1))
            kern = torch.stack(kern).view(1, -1)
            kern_grad = torch.cat(kern_grad, dim=1)
            species_matrix[species] = (kern, kern_grad)
        return species_matrix
