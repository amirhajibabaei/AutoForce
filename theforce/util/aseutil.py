#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from ase.io import Trajectory


def downsize_traj(file, l, outfile):
    t = Trajectory(file)
    tt = Trajectory(outfile, 'w')
    for k, atoms in enumerate(t):
        if k == l:
            break
        tt.write(atoms)


def make_cell_upper_triangular(atms):
    d = atms.get_all_distances()
    v = atms.get_volume()
    atms.rotate(atms.cell[2], v='z', rotate_cell=True)
    atms.rotate([*atms.cell[1][:2], 0], v='y', rotate_cell=True)
    dd = atms.get_all_distances()
    vv = atms.get_volume()
    assert np.allclose(d, dd)
    assert np.allclose(v, vv)
    assert np.allclose(atms.cell.flat[[3, 6, 7]], 0)
    atms.cell.flat[[3, 6, 7]] = 0

