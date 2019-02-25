
# coding: utf-8

# In[ ]:


import numpy as np
from ase import neighborlist


def atoms_cartesian_envs( atoms, cutoff, random=None ):
    """ 
    atoms:     ase atoms object
    cutoff:    cutoff radius
    random:    if random=int(k), k atoms will be chosen randomly
    Returns:   data
    --------------------------
    usage:
    for z1, z2, d, f in data: ...
    """
    I, J, D = neighborlist.neighbor_list( 'ijD', atoms, cutoff )
    crd = np.bincount(I)
    idx = np.cumsum(crd)[:-1]
    #i   = np.split(I,idx)
    j   = np.split(J,idx)
    d   = np.split(D,idx)
    z   = atoms.get_atomic_numbers()
    f   = atoms.get_forces()
    na  = len(atoms)
    if random is not None:
        c = np.random.choice( range(na), random )
    else:
        c = range(na)
    return [ (z[k], z[j[k]], d[k], f[k]) for k in c ]

