# +
from torch import Tensor
from typing import Tuple, Dict, Union


Key_t = Union[int, Tuple[int, ...]]
Descriptor_t = Dict[Key_t, Tensor]
