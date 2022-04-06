# +
from ..dataclass import LocalEnv
from ..function import Function
from .chempar import ChemPar
from torch import Tensor
from typing import Optional, Dict


class Cutoff(ChemPar):

    def __init__(self,
                 default: float,
                 values: Optional[Dict[int, float]] = None
                 ) -> None:
        super().__init__(default=default,
                         values=values,
                         clone=True,
                         keylen=2,
                         permsym=True,
                         bijection=None,
                         requires_grad=False
                         )
