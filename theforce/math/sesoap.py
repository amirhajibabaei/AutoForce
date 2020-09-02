# +
from theforce.util.util import iterable
from theforce.math.ylm import Ylm
import torch
from math import factorial as fac
from math import pi


Y00 = torch.tensor(pi).mul(4).sqrt().pow(-1)


class Radii:

    def __init__(self):
        pass

    def __call__(self, numbers):
        return torch.cat([torch.as_tensor(self.get(int(num))).view(1)
                          for num in numbers])

    def __repr__(self):
        return f'{self.__class__.__name__}({self.state_args})'

    def get(self, number):
        raise NotImplementedError('implement this in a subclass')


class UniformRadii(Radii):

    def __init__(self, value=1.):
        self.value = value

    def get(self, number):
        return self.value

    @property
    def state_args(self):
        return str(float(self.value))


class RadiiFromDict(Radii):

    def __init__(self, d):
        self.d = d

    def get(self, number):
        return self.d[number]

    @property
    def state_args(self):
        return str({z: float(r) for z, r in self.d.items()})


class SeSoap:

    def __init__(self, lmax, nmax, radial, radii=1., flatten=True, normalize=True):
        super().__init__()
        self.ylm = Ylm(lmax)
        self.nmax = nmax
        self.radial = radial
        if type(radii) == float:
            self.radii = UniformRadii(radii)
        elif type(radii) == dict:
            self.radii = RadiiFromDict(radii)
        else:
            self.radii = radii
        one = torch.ones(lmax+1, lmax+1)
        self.Yr = 2*torch.torch.tril(one) - torch.eye(lmax+1)
        self.Yi = 2*torch.torch.triu(one, diagonal=1)
        a = torch.tensor([[1./((2*l+1)*2**(2*n+l)*fac(n)*fac(n+l))
                           for l in range(lmax+1)] for n in range(nmax+1)])
        self.nnl = (a[None]*a[:, None]).sqrt()
        self._shape = (nmax+1, nmax+1, lmax+1)
        self.dim = (nmax+1)*(nmax+1)*(lmax+1)
        self.flatten = flatten
        if flatten:
            self._shape = (self.dim,)
        self._size = (119, 119, *self._shape)
        self.normalize = normalize
        self.params = []

    @property
    def state_args(self):
        return "{}, {}, {}, radii={}, flatten={}, normalize={}".format(
            self.ylm.lmax, self.nmax, self.radial.state, self.radii, self.flatten,
            self.normalize)

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

    def __repr__(self):
        return self.state

    def __call__(self, coo, numbers, central=None, grad=False, normalize=None, sparse_tensor=True):
        units = self.radii(numbers)
        species = torch.unique(numbers, sorted=True)
        dim0 = len(species)**2
        bcasted = torch.broadcast_tensors(species[None, ], species[:, None])
        ab = torch.cat([x.reshape(1, -1) for x in bcasted])  # alpha, beta
        xyz = coo/units.view(-1, 1)
        d = xyz.pow(2).sum(dim=-1).sqrt()
        n = 2*torch.arange(self.nmax+1).type(xyz.type())
        #
        r, dr = self.radial(units*d)
        exp = (-0.5*d**2).exp()
        if grad:
            dr = units*dr
            dexp = -d*exp
            dr = dr*exp + r*dexp
        r = r*exp
        #
        f = (r*d[None]**n[:, None])
        Y = self.ylm(xyz, grad=grad)
        if grad:
            Y, dY = Y
        ff = f[:, None, None]*Y[None]
        i = torch.arange(r.size(0))
        c = []
        for num in species:
            t = torch.index_select(ff, -1, i[numbers == num]).sum(dim=-1)
            if num == central:
                t[0, 0, 0] += Y00
            c += [t]
        c = torch.stack(c)
        nnp = c[None, :, None, ]*c[:, None, :, None]
        p = (nnp*self.Yr).sum(dim=-1) + (nnp*self.Yi).sum(dim=-2)
        if grad:
            df = dr*d[None]**n[:, None] + r*n[:, None]*d[None]**(n[:, None]-1)
            df = df[..., None]*xyz/d[:, None]
            dc = (df[:, None, None]*Y[None, ..., None] +
                  f[:, None, None, :, None]*dY[None])
            dc = torch.stack([(numbers == num).type(r.type())[:, None] * dc
                              for num in species])
            dnnp = (c[None, :, None, ..., None, None]*dc[:, None, :, None] +
                    dc[None, :, None, ]*c[:, None, :, None, ..., None, None])
            dp = ((dnnp*self.Yr[..., None, None]).sum(dim=-3) +
                  (dnnp*self.Yi[..., None, None]).sum(dim=-4))
            p, dp = p*self.nnl, dp*self.nnl[..., None, None]/units.view(-1, 1)
            if (self.normalize if normalize is None else normalize):
                norm = p.norm() + torch.finfo().eps
                p = p/norm
                dp = dp/norm
                dp = dp - p[..., None, None]*(p[..., None, None]*dp
                                              ).sum(dim=(0, 1, 2, 3, 4))
            p = p.view(dim0, *self._shape)
            dp = dp.view(dim0, *self._shape, *xyz.size())
            if sparse_tensor:
                p = torch.sparse_coo_tensor(ab, p, size=self._size)
                dp = torch.sparse_coo_tensor(
                    ab, dp, size=(*self._size, *xyz.size()))
                return p, dp
            else:
                return ab, p, self._size, dp, (*self._size, *xyz.size())
        else:
            p = p*self.nnl
            if (self.normalize if normalize is None else normalize):
                norm = p.norm() + torch.finfo().eps
                p = p/norm
            if sparse_tensor:
                p = torch.sparse_coo_tensor(ab, p.view(dim0, *self._shape),
                                            size=self._size)
                return p
            else:
                return ab, p.view(dim0, *self._shape), self._size


class SubSeSoap:

    def __init__(self, lmax, nmax, radial, numbers, radii=1., flatten=True, normalize=True):
        super().__init__()
        self.ylm = Ylm(lmax)
        self.nmax = nmax
        self.radial = radial
        if type(radii) == float:
            self.radii = UniformRadii(radii)
        elif type(radii) == dict:
            self.radii = RadiiFromDict(radii)
        else:
            self.radii = radii
        self.numbers = sorted(iterable(numbers))
        self.species = len(self.numbers)
        one = torch.ones(lmax+1, lmax+1)
        self.Yr = 2*torch.torch.tril(one) - torch.eye(lmax+1)
        self.Yi = 2*torch.torch.triu(one, diagonal=1)
        a = torch.tensor([[1./((2*l+1)*2**(2*n+l)*fac(n)*fac(n+l))
                           for l in range(lmax+1)] for n in range(nmax+1)])
        self.nnl = (a[None]*a[:, None]).sqrt()
        self.dim = self.species**2 * (nmax+1)**2 * (lmax+1)
        self.shape = (self.species, self.species, nmax+1, nmax+1, lmax+1)
        if flatten:
            self.shape = (self.dim,)
        self.flatten = flatten
        self.normalize = normalize
        self.params = []

    @property
    def state_args(self):
        return "{}, {}, {}, {}, radii={}, flatten={}, normalize={}".format(
            self.ylm.lmax, self.nmax, self.radial.state, self.numbers, self.radii,
            self.flatten, self.normalize)

    @property
    def state(self):
        return self.__class__.__name__+'({})'.format(self.state_args)

    def __repr__(self):
        return self.state

    def __call__(self, coo, numbers, central=None, grad=False, normalize=None):
        units = self.radii(numbers)
        xyz = coo/units.view(-1, 1)
        d = xyz.pow(2).sum(dim=-1).sqrt()
        n = 2*torch.arange(self.nmax+1).type(xyz.type())
        #
        r, dr = self.radial(units*d)
        exp = (-0.5*d**2).exp()
        if grad:
            dr = units*dr
            dexp = -d*exp
            dr = dr*exp + r*dexp
        r = r*exp
        #
        f = (r*d[None]**n[:, None])
        Y = self.ylm(xyz, grad=grad)
        if grad:
            Y, dY = Y
        ff = f[:, None, None]*Y[None]
        i = torch.arange(r.size(0))
        c = []
        for num in self.numbers:
            t = torch.index_select(ff, -1, i[numbers == num]).sum(dim=-1)
            if num == central:
                t[0, 0, 0] += Y00
            c += [t]
        c = torch.stack(c)
        nnp = c[None, :, None, ]*c[:, None, :, None]
        p = (nnp*self.Yr).sum(dim=-1) + (nnp*self.Yi).sum(dim=-2)
        if grad:
            df = dr*d[None]**n[:, None] + r*n[:, None]*d[None]**(n[:, None]-1)
            df = df[..., None]*xyz/d[:, None]
            dc = (df[:, None, None]*Y[None, ..., None] +
                  f[:, None, None, :, None]*dY[None])
            dc = torch.stack([(numbers == num).type(r.type())[:, None] * dc
                              for num in self.numbers])
            dnnp = (c[None, :, None, ..., None, None]*dc[:, None, :, None] +
                    dc[None, :, None, ]*c[:, None, :, None, ..., None, None])
            dp = ((dnnp*self.Yr[..., None, None]).sum(dim=-3) +
                  (dnnp*self.Yi[..., None, None]).sum(dim=-4))
            p, dp = p*self.nnl, dp*self.nnl[..., None, None]/units.view(-1, 1)
            if (self.normalize if normalize is None else normalize):
                norm = p.norm() + torch.finfo().eps
                p = p/norm
                dp = dp/norm
                dp = dp - p[..., None, None]*(p[..., None, None]*dp
                                              ).sum(dim=(0, 1, 2, 3, 4))
            return p.view(*self.shape), dp.view(*self.shape, *xyz.size())
        else:
            p = p*self.nnl
            if (self.normalize if normalize is None else normalize):
                norm = p.norm() + torch.finfo().eps
                p = p/norm
            return p.view(*self.shape)


def test_SeSoap():
    import torch
    from theforce.math.cutoff import PolyCut
    from theforce.math.soap import RealSeriesSoap

    xyz = (torch.rand(10, 3) - 0.5) * 5
    xyz.requires_grad = True
    radii = {10: 0.8, 11: 1., 18: 1.2, 19: 1.4}
    s = SeSoap(3, 3, PolyCut(8.0), radii=radii, flatten=True)
    numbers = torch.tensor(4*[10]+6*[18])

    # test grad
    p, dp = s(xyz, numbers, grad=True)
    torch.sparse.sum(p).backward()
    print('fits gradients calculated by autograd: {}'.format(
        xyz.grad.allclose(torch.sparse.sum(dp, dim=(0, 1, 2)))))

    # test non-overlapping
    numbers = torch.tensor(4*[11]+6*[19])
    pp = s(xyz, numbers, grad=False)
    t = torch.sparse.sum(p*pp).isclose(torch.tensor(0.0))
    print(f'non-overlapping: {t}')


def test_SubSeSoap():
    import torch
    from theforce.math.cutoff import PolyCut
    from theforce.math.soap import RealSeriesSoap

    xyz = (torch.rand(10, 3) - 0.5) * 5
    xyz.requires_grad = True
    radii = {10: 0.8, 11: 1., 18: 1.2, 19: 1.4}
    s = SubSeSoap(3, 3, PolyCut(8.0), [10, 18], radii=radii, flatten=True)
    numbers = torch.tensor(4*[10]+6*[18])

    # test grad
    p, dp = s(xyz, numbers, grad=True)
    p.sum().backward()
    print('fits gradients calculated by autograd: {}'.format(
        xyz.grad.allclose(dp.sum(dim=0))))

    # test non-overlapping
    numbers = torch.tensor(4*[11]+6*[19])
    pp = s(xyz, numbers, grad=False)
    t = torch.sum(p*pp).isclose(torch.tensor(0.0))
    print(f'non-overlapping: {t}')


if __name__ == '__main__' and True:
    test_SeSoap()
    test_SubSeSoap()
