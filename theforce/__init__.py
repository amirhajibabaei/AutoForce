__version__ = "v2021.09"

# +
import torch

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.double)
if int(torch.__version__.split(".")[1]) > 7:
    torch.cholesky = torch.linalg.cholesky
    torch.qr = torch.linalg.qr

# +
import importlib
import sys

sys.modules["theforce.math"] = importlib.import_module("theforce.descriptor")
