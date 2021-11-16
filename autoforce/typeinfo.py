# +
import torch
from math import pi as _pi


float_t = torch.float32
finfo = torch.finfo(float_t)
eps = finfo.eps
pi = torch.tensor(_pi, dtype=float_t)
