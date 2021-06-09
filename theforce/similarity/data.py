# +
import torch
from theforce.util.tensors import nan_to_num
from mendeleev import get_table


def atom_embeddings(keys=['vdw_radius', 'en_pauling', 'electron_affinity'], normalize=True):
    tab = get_table('elements')
    z = torch.tensor(tab['atomic_number'].__array__())
    e = nan_to_num(torch.tensor(tab[keys].__array__()), 0.)
    if normalize:
        e /= e.var(dim=0).sqrt()
    return z, e
