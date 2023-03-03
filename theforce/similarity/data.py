# +
import torch
from mendeleev.fetch import fetch_table

from theforce.util.tensors import nan_to_num


def atom_embeddings(
    keys=["vdw_radius", "en_pauling", "electron_affinity"], normalize=True
):
    tab = fetch_table("elements")
    z = torch.tensor(tab["atomic_number"].__array__())
    e = nan_to_num(torch.tensor(tab[keys].__array__()), 0.0)
    if normalize:
        e /= e.var(dim=0).sqrt()
    return z, e
