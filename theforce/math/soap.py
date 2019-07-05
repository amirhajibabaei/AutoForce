
# coding: utf-8

# In[ ]:


import torch
from theforce.math.ylm import Ylm
from torch.nn import Module
from math import factorial as fac


class AbsSeriesSoap(Module):

    def __init__(self, lmax, nmax, radial, unit=None):
        super().__init__()
        self.ylm = Ylm(lmax)
        self.nmax = nmax
        self.radial = radial
        if unit:
            self.unit = unit
        else:
            self.unit = radial.rc/3
        one = torch.ones(lmax+1, lmax+1)
        self.Yr = 2*torch.torch.tril(one) - torch.eye(lmax+1)
        self.Yi = 2*torch.torch.triu(one, diagonal=1)

    @property
    def state_args(self):
        return "{}, {}, {}, unit={}".format(self.ylm.lmax, self.nmax, self.radial.state, self.unit)

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

    def forward(self, coo, grad=True):
        xyz = coo/self.unit
        d = xyz.pow(2).sum(dim=-1).sqrt()
        n = 2*torch.arange(self.nmax+1).type(xyz.type())
        r, dr = self.radial(self.unit*d)
        dr = self.unit*dr
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
            return p, dp/self.unit
        else:
            return p


class SeriesSoap(Module):

    def __init__(self, lmax, nmax, radial, unit=None, modify=None, normalize=False,
                 cutcorners=0, symm=False):
        super().__init__()
        self.abs = AbsSeriesSoap(lmax, nmax, radial, unit=unit)

        if modify:
            a = torch.tensor([[modify**(2*n+l)/((2*l+1)*2**(2*n+l)*fac(n)*fac(n+l))
                               for l in range(lmax+1)] for n in range(nmax+1)])
            self.nnl = a[None]*a[:, None]
        else:
            self.nnl = torch.ones(nmax+1, nmax+1, lmax+1)

        n = torch.arange(nmax+1)
        self.mask = ((n[:, None]-n[None]).abs() <= nmax-cutcorners).byte()
        if not symm:
            self.mask = (self.mask & (n[:, None] >= n[None]).byte())

        self.normalize = normalize

        self.kwargs = 'modify={}, normalize={}, cutcorners={}, symm={}'.format(
            modify, normalize, cutcorners, symm)

    def forward(self, xyz, grad=True):
        p = self.abs(xyz, grad=grad)
        if grad:
            p, q = p
            q = q*self.nnl[..., None, None]
        p = p*self.nnl

        p = p[self.mask].view(-1)
        if grad:
            q = q[self.mask].view(p.size(0), *xyz.size())

        if self.normalize:
            norm = p.norm()
            if norm > 0.0:
                p = p/norm
                if grad:
                    q = q/norm
                    q = q - p[..., None, None] * (p[..., None, None] * q
                                                  ).sum(dim=(0))
        if grad:
            return p, q
        else:
            return p

    @property
    def dim(self):
        return self.mask.sum()*(self.abs.ylm.lmax+1)

    @property
    def state_args(self):
        return "{}, {}".format(self.abs.state_args, self.kwargs)

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)


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

    s = AbsSeriesSoap(2, 2, PolyCut(3.0))
    p, dp = s(xyz)
    p = p.permute(2, 0, 1)
    print('fits pre-calculated values: {}'.format(p.allclose(target)))

    p.sum().backward()
    print('fits gradients calculated by autograd: {}'.format(
        xyz.grad.allclose(dp.sum(dim=(0, 1, 2)))))

    # test with normalization turned on
    s = SeriesSoap(3, 7, PolyCut(3.0), normalize=True)
    xyz.grad *= 0
    p, dp = s(xyz)
    p.sum().backward()
    print('fits gradients calculated by autograd (normalize=True):{}'.format(
        xyz.grad.allclose(dp.sum(dim=(0)))))

    assert s.state == eval(s.state).state

    # test if works with empty tensors
    s(torch.rand(0, 3))


def test_units():
    from theforce.math.cutoff import PolyCut
    xyz = torch.tensor([[0.175, 0.884, -0.87, 0.354, -0.082, 3.1],
                        [-0.791, 0.116, 0.19, -0.832, 0.184, 0.],
                        [0.387, 0.761, 0.655, -0.528, 0.973, 0.]]).t()
    xyz = xyz*3
    cutoff = 3.0*3
    xyz.requires_grad = True

    s = SeriesSoap(3, 3, PolyCut(cutoff), normalize=True)
    p, dp = s(xyz)
    p.sum().backward()
    print('grads are consistent with larger length scale: {}'.format(
        xyz.grad.allclose(dp.sum(dim=(0)))))


def test_speed(N=100):
    from theforce.math.cutoff import PolyCut
    import time
    s = AbsSeriesSoap(5, 5, PolyCut(3.0))
    start = time.time()
    for _ in range(N):
        xyz = torch.rand(30, 3)
        p = s(xyz)
    finish = time.time()
    delta = (finish-start)/N
    print("speed of {}: {} sec".format(s.state, delta))


def example():
    from theforce.math.cutoff import PolyCut

    lengthscale = 2.
    cutoff = 8.
    xyz = torch.tensor([[1., 0, 0], [-1., 0, 0],
                        [0, 1., 0], [0, -1., 0],
                        [0, 0, 1.], [0, 0, -1.]]) * lengthscale
    xyz.requires_grad = True
    s = SeriesSoap(2, 2, PolyCut(cutoff), normalize=True)
    p, dp = s(xyz)
    print(p)


if __name__ == '__main__':
    test_validity()
    test_units()
    test_speed()

