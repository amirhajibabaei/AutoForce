# +
from torch import Tensor
from typing import Tuple, Dict, Union


Key = Union[int, Tuple[int, ...]]
TensorDict = Dict[Key, Tensor]
