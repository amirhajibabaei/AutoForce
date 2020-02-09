import torch


def coeff_of_determination(pred, target):
    p = torch.as_tensor(pred)
    t = torch.as_tensor(target)
    var1 = t.var()
    var2 = (t-p).var()
    R2 = 1-var2/var1
    return R2

