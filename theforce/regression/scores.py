# +
import numpy as np
import torch


def _as_tensor(pred, target):
    p = torch.as_tensor(pred)
    t = torch.as_tensor(target)
    return p, t


def maxe(pred, target):
    p, t = _as_tensor(pred, target)
    return (p-t).abs().max()


def mae(pred, target):
    p, t = _as_tensor(pred, target)
    return (p-t).abs().mean()


def rmse(pred, target):
    p, t = _as_tensor(pred, target)
    return (p-t).pow(2).mean().sqrt()


def cd(pred, target):
    p, t = _as_tensor(pred, target)
    var1 = t.var()
    var2 = (t-p).var()
    R2 = 1-var2/var1
    return R2


coeff_of_determination = cd


def get_energy_and_forces(data):
    d_e = []
    d_f = []
    d_n = []
    for d in data:
        d_e.append(d.get_potential_energy())
        d_f.append(d.get_forces().reshape(-1))
        d_n.append(d.numbers)
    d_e = np.stack(d_e).reshape(-1)
    d_f = np.concatenate(d_f)
    d_n = np.concatenate(d_n)
    d_n = d_n.reshape(-1, 1).repeat(3, axis=1).reshape(-1)
    return d_e, d_f, d_n


if __name__ == '__main__':
    import sys
    from ase.io import read

    data = read(sys.argv[1], '::')
    targets = read(sys.argv[2], '::')
    d_e, d_f, d_n = get_energy_and_forces(data)
    t_e, t_f, t_n = get_energy_and_forces(targets)
    assert (d_n == t_n).all()
    quant = len(data)
    assert len(targets) == quant

    def _print(d, t):
        print(f'\tmaxe: {maxe(d, t)}')
        print(f'\tmae:  {mae(d, t)}')
        print(f'\trmse: {rmse(d, t)}')
        print(f'\tcd:   {cd(d, t)}')

    print(f'predictions:       {sys.argv[1]}')
    print(f'targets:           {sys.argv[2]}')
    print(f'number of samples: {quant}')

    for n, d, t in [('energy', d_e, t_e), ('forces', d_f, t_f)]:
        print(f'\n{n}:')
        _print(d, t)
        if n == 'forces':
            zset = np.unique(d_n)
            if len(zset) > 1:
                for z in zset:
                    i = d_n == z
                    s = i.sum()//3
                    print(f'\nforces on (atomic number): {z}  (size= 3*{s})')
                    _print(d[i], t[i])
