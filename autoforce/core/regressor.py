# +
from .dataclasses import Conf, Target
from .cutoff import Cutoff
from .kernel import Kernel
from torch import Tensor
import torch
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Union, Any


class Regressor(ABC):
    """
    TODO:

    """

    @property
    @abstractmethod
    def cutoff(self) -> Union[Cutoff, None]:
        """
        TODO:

        """
        ...

    @abstractmethod
    def get_design_matrix(self, confs: Sequence[Conf]) -> (Tensor, Tensor, Any):
        """
        TODO:

        """
        ...

    @abstractmethod
    def set_weights(self, weights: Tensor, sections: Any) -> None:
        """
        TODO:

        """
        ...

    @abstractmethod
    def get_target(self, conf: Conf) -> Target:
        """
        TODO:

        """
        ...


class KernelRegressor(Regressor):
    """
    TODO:

    """

    def __init__(self, kernel: Kernel) -> None:
        """
        TODO:

        """
        self.kernel = kernel
        self.basis = kernel.descriptor.new_basis()

    @property
    def cutoff(self) -> Cutoff:
        return self.kernel.descriptor.cutoff

    def get_design_matrix(self,
                          confs: Sequence[Conf]
                          ) -> (Tensor, Tensor, Tuple[Tuple[int, int], ...]):
        """
        TODO:

        """
        sections, ke, kf = self.kernel.get_design_matrix(confs, self.basis)
        return ke, kf, sections

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
        e = self.kernel.get_potential_energy(conf, self.basis, self.weights)
        if e.grad_fn:
            g, = torch.autograd.grad(e, conf.positions, retain_graph=True)
        else:
            g = 0
        return Target(energy=e.detach(), forces=-g)
