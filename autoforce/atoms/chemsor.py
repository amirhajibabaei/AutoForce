import autoforce.cfg as cfg
import torch
from torch import Tensor
from itertools import product
from typing import Optional, Union, Dict, Tuple


class Chemsor:
    """
    Chemical + Tensor = Chemsor!

    A canonical class for treating atom specific parameters.

    Specified keys:

        >>> cutoff = Chemsor({(1, 1): 3., (1, 6): 4., (6, 6): 5})

    Default for all:

        >>> cutoff = Chemsor(default=6.)

    Default + specified keys:

        >>> cutoff = Chemsor({(1, 1): 3., (1, 6): 4., (6, 6): 5}, default=6.)

    Indices can be specified afterwards as well.
    For example, the last example is equivalent to:

        >>> cutoff = Chemsor(index_dim=2)
        >>> cutoff[:] = 6.   # sets the default value
        >>> cutoff[1, 1] = 3.
        >>> cutoff[1, 6] = 4.
        >>> cutoff[6, 6] = 6.

    If perm_sym = True, it is ensured that values are
    invariant wrt permutation of indices (x[1, 2] == x[2, 1]).

    Other dimensionality for indices are also accepted:

        >>> mass = Chemsor({1: 1., 2: 2., ...})

    Finally, it can be used as callable:

        >>> cutoff = Chemsor({(1, 1): 3., (1, 6): 4., (6, 6): 5.})
        >>> a = torch.tensor(1)
        >>> b = torch.tensor([1, 1, 6, 6])
        >>> cutoff(a, b)
        tensor([3., 3., 4., 4.])

        >>> cutoff(b, b)
        tensor([3., 3., 5., 5.])

    """

    def __init__(self,
                 dict: Optional[Dict[Tuple[int, ...], Tensor]] = None,
                 default: Optional[Tensor] = None,
                 index_dim: Optional[int] = None,
                 perm_sym: Optional[bool] = True
                 ) -> None:

        self.index_dim = index_dim
        self.perm_sym = perm_sym
        self.dict = {}
        self[:] = default

        if dict is None:
            if self.index_dim is None:
                raise RuntimeError('Since dict=None, index_dim can not be inferred! '
                                   'Specify index_dim explicitly!')
        else:
            for key, val in dict.items():
                if self.index_dim is None:
                    if hasattr(key, '__iter__'):
                        self.index_dim = len(key)
                    else:
                        self.index_dim = 1
                self[self._getkey(key)] = val

    def _getkey(self, key: Tuple[int, ...]) -> Tuple[int, ...]:

        if not hasattr(key, '__iter__'):
            key = (key,)

        if len(key) != self.index_dim:
            raise IndexError(
                f'key {key} has wrong length! (!={self.index_dim})')

        if self.perm_sym:
            key = tuple(sorted(key))
        else:
            key = tuple(key)

        return key

    def __getitem__(self, key: Tuple[int, ...]) -> Union[Tensor, None]:
        key = self._getkey(key)
        if key in self.dict:
            return self.dict[key]
        else:
            return self.default

    def __setitem__(self, key: Tuple[int, ...], value: Tensor) -> None:
        if value is not None:
            value = torch.as_tensor(value).to(cfg.float_t)
        if type(key) == slice:
            if all([x is None for x in (key.start, key.stop, key.step)]):
                self.dict = {}
                self.default = value
            else:
                raise RuntimeError('Only [:] slice can be specified!')
        else:
            self.dict[self._getkey(key)] = value

    def as_dict(self, species: Tuple[int]) -> Dict[Tuple[int], Tensor]:

        if not hasattr(species, '__iter__'):
            species = (species,)

        species = set(species)
        dict = {}
        for i in product(species, repeat=self.index_dim):
            if self.perm_sym:
                i = tuple(sorted(i))
            val = self[i]
            if val is not None:
                dict[i] = val

        return dict

    def __call__(self, *args: Tuple[Tensor, ...]) -> Tensor:
        """
        args are a tuple (length=index_dim) of broadcastable tensors.

        """
        args = torch.broadcast_tensors(*args)
        keys = torch.stack(args).t().tolist()
        result = torch.stack([self[key] for key in keys])
        return result


def test_Chemsor() -> bool:
    #
    s = Chemsor({1: 1.})
    assert s[(1,)] == s[1] == 1. and s[2] == None
    s[:] = 3.
    assert s[1] == s[2] == 3.

    #
    s = Chemsor({1: 1.}, 3.)
    assert s[1] == 1. and s[54345] == 3.
    a = torch.tensor([1, 2])
    assert s(a).allclose(torch.tensor([1., 3.]))

    #
    s = Chemsor({(1, 2): 1.})
    assert s[1, 2] == s[2, 1] == s[(1, 2)] == s[(
        2, 1)] == 1. and s[1, 3] == None
    s[:] = 2.
    assert s[117, 75] == 2.

    #
    s = Chemsor({(1, 2): 1.}, perm_sym=False)
    assert s[1, 2] == 1. and s[2, 1] == None

    #
    s = Chemsor({(1, 2): 1.}, 3., perm_sym=False)
    assert s[1, 2] == 1. and s[2, 1] == 3.

    #
    s = Chemsor({(1, 2): 1., (2, 1): 2.}, 3., perm_sym=False)
    assert s[1, 2] == 1. and s[2, 1] == 2.

    #
    s = Chemsor(index_dim=2)
    s[:] = 1.
    s[1, 2] = 2.
    assert s[1, 1] == s[2, 2] == 1. and s[1, 2] == s[2, 1] == 2.

    #
    s = Chemsor({(1, 2): 1., (2, 2): 7.}, 3.)
    a = torch.tensor(1)
    b = torch.tensor([1, 2, 3])
    assert (s(a, b) == torch.tensor([3., 1., 3.])).all()
    assert (s(b, b) == torch.tensor([3., 7., 3.])).all()

    return True


if __name__ == '__main__':
    print(test_Chemsor())