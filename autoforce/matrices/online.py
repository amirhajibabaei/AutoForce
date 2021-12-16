# +
import torch
from torch import Tensor
from typing import List, Optional, Any
from abc import ABC, abstractmethod


class OnlineABC(ABC):

    def __init__(self, data: Optional[Tensor] = None) -> None:
        self.data = data

    @property
    def size(self) -> torch.Size:

        if self.data is None:
            return torch.Size((0, 0))
        else:
            return self.data.size()

    @abstractmethod
    def append_(self, vec: Tensor, *args: Any, **kwargs: Any) -> bool:
        ...


class OnlineMatrix(OnlineABC):

    def __init__(self, data: Optional[Tensor] = None) -> None:
        super().__init__(data)

    def append_(self, vec: Tensor, dim: Optional[int] = 0) -> bool:
        """
        dim = 0 -> add vec as row
        dim = 1 -> add vec as col

        """

        if dim == 0:
            vec = vec.view(1, -1)
        elif dim == 1:
            vec = vec.view(-1, 1)
        else:
            raise RuntimeError("dim sould be either 1 or 2!")

        if self.data is None:
            self.data = vec
        else:
            self.data = torch.cat([self.data, vec], dim=dim)

        return True


class OnlineSymMatrix(OnlineMatrix):

    def __init__(self, data: Optional[Tensor] = None) -> None:
        super().__init__(data)
        if self.data is None:
            self.data = torch.empty(0, 0)

    def append_(self, vec: Tensor, diag: Optional[Tensor] = None) -> bool:

        if diag is None:
            diag = vec[-1]
            vec = vec[:-1]
        diag = diag.view(1, 1)
        vec = vec.view(-1, 1)

        self.data = torch.cat([torch.cat([self.data, vec], dim=1),
                               torch.cat([vec.t(), diag], dim=1)])
        return True


class OnlineTriMatrix(OnlineMatrix):

    def __init__(self, data: Optional[Tensor] = None, upper=False) -> None:
        super().__init__(data)
        self.upper = upper
        if self.data is None:
            self.data = torch.empty(0, 0)

    def append_(self, vec: Tensor, diag: Optional[Tensor] = None) -> bool:

        if diag is None:
            diag = vec[-1]
            vec = vec[:-1]
        diag = diag.view(1, 1)
        vec = vec.view(-1, 1)

        if self.upper:
            a = vec
            b = torch.zeros_like(vec).t()
        else:
            a = torch.zeros_like(vec)
            b = vec.t()

        self.data = torch.cat([torch.cat([self.data, a], dim=1),
                               torch.cat([b, diag], dim=1)])
        return True


class OnlineSPD(OnlineSymMatrix):

    def __init__(self, data: Optional[Tensor] = None) -> None:
        super().__init__(data)
        self._chol = None
        if self.size[0] > 0:
            self._invert()

    def _invert(self) -> None:
        self._chol = torch.linalg.cholesky(self.data)
        self._inv = self._chol.cholesky_inverse()

    def append_(self,
                vec: Tensor,
                diag: Optional[Tensor] = None,
                spilling: Optional[Tensor] = None
                ) -> bool:

        if self.size[0] == 0 or spilling is None:
            app = True
        else:

            if diag is None:
                diag = vec[-1]
                vec = vec[:-1]
            diag = diag.view(1, 1)

            vec = vec.view(-1, 1)
            alpha = (diag - vec.t()@self._inv@vec)/diag

            if alpha > spilling:
                app = True
            else:
                app = False

        if app:
            super().append_(vec, diag=diag)
            self._invert()

        return app


def test_OnlineMatrix():

    a = torch.rand(5, 10)

    M = OnlineMatrix()
    for i in range(5):
        M.append_(a[i], dim=0)
    test1 = M.data.allclose(a)

    N = OnlineMatrix()
    for i in range(10):
        N.append_(a[:, i], dim=1)
    test2 = N.data.allclose(a)

    return test1, test2


def test_OnlineTriMatrix():

    a = torch.rand(5, 5)
    l = torch.tril(a)
    u = torch.triu(a)

    M = OnlineTriMatrix()
    N = OnlineTriMatrix(upper=True)
    for i in range(5):
        M.append_(a[i, :i+1])
        N.append_(a[:i+1, i])
    test1 = M.data.allclose(l)
    test2 = N.data.allclose(u)

    return test1, test2


def test_OnlineSymMatrix():

    a = torch.rand(5, 5)
    a = a + a.t()

    M = OnlineSymMatrix()
    N = OnlineSymMatrix()
    for i in range(5):
        M.append_(a[i, :i+1])
        N.append_(a[i, :i], diag=a[i, i])
    test1 = M.data.allclose(a)
    test2 = N.data.allclose(a)

    return test1, test2


def test_OnlineSPD():

    Xind = torch.tensor([0.])
    m = OnlineSPD(torch.ones(1, 1))
    Xdat = torch.arange(0, 5, 0.01)

    for x in Xdat:
        k = x.sub(Xind).pow(2).mul(-0.5).exp()
        app = m.append_(k, diag=torch.ones([]), spilling=1e-1)
        if app:
            Xind = torch.cat([Xind, x.view(1)])

    test = ((m._inv@m.data)-torch.eye(m.size[0])).abs().max().lt(1e-4)

    return bool(test)


if __name__ == '__main__':
    a = test_OnlineMatrix()
    b = test_OnlineTriMatrix()
    c = test_OnlineSymMatrix()
    d = test_OnlineSPD()
    print(a, b, c, d)
