# +
"""
Global configuration (dtype, constants, etc.)
of autoforce.

The default dtype is torch.float32 which is
chosen over float64 for the sake of memory
efficiency.

If desired, the dfault dtype can be changed

    >>> import autoforce.cfg as cfg
    >>> import torch
    >>> cfg.configure(torch.float64)
   
   
Note:

Internally, we explicitly set the dtype of
every tensor created by

    torch.tensor(..., dtype=cfg.float_t)
    torch.zeros(..., dtype=cfg.float_t)
    etc.
    
We did not use

    torch.set_default_dtype(cfg.float_t)
    
because this package should not change
the global status of other packages.

"""
import torch
from math import pi as _pi

# Defaults:
float_t = torch.float32

# Constants
finfo = torch.finfo(float_t)
eps = finfo.eps
zero = torch.tensor(0.0, dtype=float_t)
one = torch.tensor(1.0, dtype=float_t)
pi = torch.tensor(_pi, dtype=float_t)


def configure(dtype: torch.dtype) -> None:
    global float_t, finfo, eps, pi
    float_t = dtype
    finfo = torch.finfo(float_t)
    eps = finfo.eps
    pi = torch.tensor(_pi, dtype=float_t)
