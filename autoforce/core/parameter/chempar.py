# +
import autoforce.cfg as cfg
from ..function import Bijection
from torch import Tensor
import torch
from itertools import product
from typing import Optional, Union, Dict, Tuple, Callable, Any


class ChemPar:
    """
    A class for broadcasting species dependent
    parameters. For example the pair interaction
    cutoff can depend on the pair type:
    >>> R[a, b] = x
    Arbitrary number of indices (keylen) are
    possible: R[a, b, c, ...] = x. 
    Since specification of all key-value pairs
    can become cumbersome and in some cases all
    possible keys are not clear beforehand, one
    can also define a default value for all
    possible future keys. For example:

    >>> values = {(1, 1): 3.,
                  (1, 6): 4.,
                  (6, 6): 5.}
    >>> R = ChemPar(values=values, default=7.)

    Here, if a key e.g. (1, 7) is absent from
    the values dict, the default value of 7.
    is assumed. Additionally, if clone = True
    the new keys are explicitly added to the
    values dict. This is only significant during
    parameter optimization. With clone = False,
    the default value is shared between all
    the absent keys and is bound to remain the
    same. While with clone = True, each key
    has its own parameter.

    If permsym = True, it is ensured that the
    values are invariant wrt permutation of
    indices: R[6, 1] == R[1, 6].

    It can be used for broadcasting as a
    callable:

    >>> a = torch.tensor(1)
    >>> b = torch.tensor([1, 1, 6, 6])
    >>> R(a, b)
    tensor([3., 3., 4., 4.])
    >>> R(b, b)
    tensor([3., 3., 5., 5.])

    Constraints can be enforced using a
    bijection function. In this case
    bijection.inverse is called for setting
    the values and bijection.forward is
    called when the values are requested.

    """

    __slots__ = ('values', 'default', 'clone', 'keylen',
                 'permsym', 'bijection', '_requires_grad')

    def __init__(self,
                 *,
                 values: Optional[Dict[Tuple[int, ...], Tensor]] = None,
                 default: Optional[Tensor] = None,
                 clone: Optional[bool] = True,
                 keylen: Optional[int] = None,
                 permsym: Optional[bool] = True,
                 bijection: Optional[Bijection] = None,
                 requires_grad: Optional[bool] = False
                 ) -> None:
        """
        values         a key-value dictionary
        default        the value for future keys which
                       absent from values
        clone          if True, it adds new keys to the
                       values dict and sets them equal to
                       cloned default.
        keylen         length of the keys, only needed if
                       values is None
        permsym        if True, the parameter is invariant
                       wrt permutation os indices
        bijection      applies a bijective transformation
        requires_grad  turns on gradient tracing

        """

        self.clone = clone
        self.keylen = keylen
        self.permsym = permsym
        self.bijection = bijection
        self.values = {}
        self[:] = default

        if values is None or values is {}:
            if self.keylen is None:
                raise RuntimeError('Specify keylen explicitly!')
        else:
            for key, val in values.items():
                if self.keylen is None:
                    if hasattr(key, '__iter__'):
                        self.keylen = len(key)
                    else:
                        self.keylen = 1
                self[self._getkey(key)] = val

        self.requires_grad = requires_grad

    def parameters(self) -> Tuple[Tensor]:
        if self.default is None or self.clone:
            return (*self.values.values(),)
        else:
            return (self.default, *self.values.values())

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value
        for par in self.parameters():
            par.requires_grad = value

    def _getkey(self, key: Tuple[int, ...]) -> Tuple[int, ...]:

        if not hasattr(key, '__iter__'):
            key = (key,)

        if len(key) != self.keylen:
            raise IndexError(f'The key {key} has the wrong'
                             ' length! (!={self.keylen})')

        if self.permsym:
            key = tuple(sorted(key))
        else:
            key = tuple(key)

        return key

    def __setitem__(self, key: Tuple[int, ...], value: Tensor) -> None:

        if value is not None:
            value = torch.as_tensor(value).to(cfg.float_t)

            if self.bijection:
                value = self.bijection.inverse(value)

        if type(key) == slice:
            if all([x is None for x in (key.start, key.stop, key.step)]):
                self.values = {}
                self.default = value
            else:
                raise RuntimeError('Only [:] slice can be specified!')
        else:
            self.values[self._getkey(key)] = value

    def __getitem__(self, key: Tuple[int, ...]) -> Union[Tensor, None]:

        key = self._getkey(key)
        if key in self.values:
            value = self.values[key]
        else:
            if self.clone:
                value = self.default
                if value is not None:
                    value = value.clone().detach()
                    value.requires_grad = self.requires_grad
                self.values[key] = value
            else:
                value = self.default

        if self.bijection:
            value = self.bijection.forward(value)

        return value

    def as_dict(self,
                species: Tuple[int],
                convert: Optional[Callable] = None
                ) -> Dict[Tuple[int], Tensor]:

        if not hasattr(species, '__iter__'):
            species = (species,)

        species = set(species)
        values = {}
        for i in product(species, repeat=self.keylen):
            if self.permsym:
                i = tuple(sorted(i))
            val = self[i]
            if val is not None:
                if convert:
                    val = convert(val)
                values[i] = val

        return values

    def __call__(self, *args: Tuple[Tensor, ...]) -> Tensor:
        """
        args = (..., tj, ...)

        where tj are int tensors,
        length(args) == keylen,
        and all tensors have the same size or
        can be broadcasted to the same size.

        """
        args = torch.broadcast_tensors(*args)
        keys = torch.stack(args).t().tolist()
        result = torch.stack([self[key] for key in keys])
        return result


class ReducedPar:

    def __init__(self,
                 *par: Any,
                 op: Optional[Callable] = max
                 ) -> None:
        self.par = par
        self.op = op

    def include(self, *par: Any) -> None:
        self.par = (*self.par, *par)

    def as_dict(self,
                species: Tuple[int],
                convert: Optional[Callable] = None
                ) -> Dict:
        dct = {}
        for par in self.par:
            _dct = par.as_dict(species, convert).items()
            for k, v in _dct:
                if k in dct:
                    dct[k] = self.op(dct[k], v)
                else:
                    dct[k] = v
        return dct
