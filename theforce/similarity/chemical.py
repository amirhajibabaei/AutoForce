# +
import torch
from theforce.similarity.universal import UniversalSoapKernel, Chemical
from theforce.util.util import iterable
from theforce.util.data import atom_embeddings


class ChemRBF(Chemical):

    def __init__(self, embeddings=None):
        super().__init__()
        if embeddings is None:
            z, em = atom_embeddings()
            self.em = torch.zeros(119, em.size(1))
            self.em[z] += em
        else:
            self.em = embeddings
        self.em.requires_grad = True
        self.params = [self.em]

    def forward(self, _a, _b):
        a = torch.as_tensor(_a).view(-1)
        b = torch.as_tensor(_b).view(-1)
        k = self.em[a][:, None]-self.em[b][None]
        return k.pow(2).sum(dim=-1).neg().exp()


class ChemicalSoapKernel(UniversalSoapKernel):

    def __init__(self, *args, **kwargs):
        if 'chemical' in kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, chemical=ChemRBF(), **kwargs)

    def get_func(self, _p, _q):
        c = torch.tensor(0.)
        for p in iterable(_p):
            d = self.saved(p, 'value')
            if d is None:
                continue
            for q in iterable(_q):
                mu = self.kern(p.number, q.number)
                if mu > 0.:
                    dd = self.saved(q, 'value')
                    if dd is None:
                        continue
                    ab1, x1, _ = d
                    ab2, x2, _ = dd
                    c += ((x1[:, None]*x2[None]).sum(dim=-1) *
                          self.kern(ab1[0],  ab2[0]) *
                          self.kern(ab1[1],  ab2[1])
                          ).sum()**self.exponent*mu.view([])
        return c.view(1, 1)
