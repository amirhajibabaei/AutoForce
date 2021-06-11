__version__ = 'v2021.05'

# +
import torch

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.double)

# +
import importlib
import sys

sys.modules['theforce.math'] = importlib.import_module('theforce.descriptor')
