
# coding: utf-8

# In[ ]:


import torch


def cat(tensors, dim=0):
    lengths = [tensor.size(dim) for tensor in tensors]
    cat = torch.cat(tensors, dim=dim)
    spec = torch.LongTensor(lengths + [dim])
    return cat, spec


def split(tensor, spec):
    return torch.split(tensor, spec[:-1].tolist(), spec[-1])


# -------------------------------------------------------------
def test():
    # cat and split
    a = torch.rand(10, 7, 3)
    b = torch.rand(10, 8, 3)
    c = torch.rand(10, 9, 3)
    t, spec = cat([a, b, c], 1)
    print([a.shape for a in split(t, spec)])


if __name__ == '__main__':
    test()

