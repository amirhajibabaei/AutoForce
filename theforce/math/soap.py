
# coding: utf-8

# In[ ]:


import torch
from theforce.math.ylm import Ylm
from torch.nn import Module


class SeriesSoap(Module):

    def __init__(self, lmax, nmax, radial):
        super().__init__()
        self.ylm = Ylm(lmax)
        self.nmax = nmax
        self.radial = radial
        one = torch.ones(lmax+1, lmax+1)
        self.Yr = 2*torch.torch.tril(one) - torch.eye(lmax+1)
        self.Yi = 2*torch.torch.triu(one, diagonal=1)

    @property
    def state_args(self):
        return "{}, {}, {}".format(self.ylm.lmax, self.nmax, self.radial.state)

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

    def forward(self, xyz, grad=True):
        d = xyz.pow(2).sum(dim=-1).sqrt()
        n = 2*torch.arange(self.nmax+1).type(xyz.type())
        r, dr = self.radial(d)
        f = (r*d[None]**n[:, None])
        Y = self.ylm(xyz, grad=grad)
        if grad:
            Y, dY = Y
        c = (f[:, None, None]*Y[None]).sum(dim=-1)
        nnp = c[None, ]*c[:, None]
        p = (nnp*self.Yr).sum(dim=-1) + (nnp*self.Yi).sum(dim=-2)
        if grad:
            df = dr*d[None]**n[:, None] + r*n[:, None]*d[None]**(n[:, None]-1)
            df = df[..., None]*xyz/d[:, None]
            dc = (df[:, None, None]*Y[None, ..., None] +
                  f[:, None, None, :, None]*dY[None])
            dnnp = (c[None, ..., None, None]*dc[:, None] +
                    dc[None, ]*c[:, None, ..., None, None])
            dp = ((dnnp*self.Yr[..., None, None]).sum(dim=-3) +
                  (dnnp*self.Yi[..., None, None]).sum(dim=-4))
            return p.permute(2, 0, 1), dp.permute(2, 0, 1, 3, 4)
        else:
            return p.permute(2, 0, 1)


def test_validity():
    import torch
    from theforce.math.cutoff import PolyCut

    xyz = torch.tensor([[0.175, 0.884, -0.87, 0.354, -0.082, 3.1],
                        [-0.791, 0.116, 0.19, -0.832, 0.184, 0.],
                        [0.387, 0.761, 0.655, -0.528, 0.973, 0.]]).t()
    xyz.requires_grad = True

    target = torch.tensor([[[0.36174603, 0.39013356, 0.43448023],
                            [0.39013356, 0.42074877, 0.46857549],
                            [0.43448023, 0.46857549, 0.5218387]],

                           [[0.2906253, 0.30558356, 0.33600938],
                            [0.30558356, 0.3246583, 0.36077952],
                            [0.33600938, 0.36077952, 0.40524778]],

                           [[0.16241845, 0.18307552, 0.20443194],
                            [0.18307552, 0.22340802, 0.26811937],
                            [0.20443194, 0.26811937, 0.34109511]]])

    s = SeriesSoap(2, 2, PolyCut(3.0))
    p, dp = s(xyz)
    print('fits pre-calculated values: {}'.format(p.allclose(target)))

    p.sum().backward()
    print('fits gradients calculated by autograd: {}'.format(
        xyz.grad.allclose(dp.sum(dim=(0, 1, 2)))))


def test_speed(N=100):
    from theforce.math.cutoff import PolyCut
    import time
    s = SeriesSoap(5, 5, PolyCut(3.0))
    start = time.time()
    for _ in range(N):
        xyz = torch.rand(30, 3)
        p = s(xyz)
    finish = time.time()
    delta = (finish-start)/N
    print("speed of {}: {} sec".format(s.state, delta))


if __name__ == '__main__':
    test_validity()
    test_speed()

