# +
"""
Global configuration of autoforce.


*** Precision ***

The default precision is set by: dtype=float64.

If desired, the precision can be globally set by:
    >>> import autoforce.cfg as cfg
    >>> import torch
    >>> cfg.configure_precision(torch.floatXX)


Note 1: Internally, we explicitly set the dtype
of every new tensor by
    >>> torch.tensor(..., dtype=cfg.dtype)
    >>> torch.zeros(..., dtype=cfg.dtype)
    etc.
We do not use
    >>> torch.set_default_dtype(cfg.dtype)
because this package should not change the global
status of other packages.


Note 2: Although float32 is preferred for memory
efficiency, it dramatically undermines accuracy.
If this lack of accuracy is only related to algebraic
operations, a mixed precision scheme may be adopted
in future.

"""
import torch
from math import pi as _pi


def configure_precision(dtype: torch.dtype) -> None:
    """
    dtype: torch.float64 or torch.float32
    """

    glob = globals()
    glob['float_t'] = dtype
    glob['finfo'] = torch.finfo(dtype)
    glob['eps'] = finfo.eps
    glob['zero'] = torch.tensor(0.0, dtype=dtype)
    glob['one'] = torch.tensor(1.0, dtype=dtype)
    glob['empty'] = torch.empty(0, dtype=dtype)
    glob['pi'] = torch.tensor(_pi, dtype=dtype)


configure_precision(torch.float64)
