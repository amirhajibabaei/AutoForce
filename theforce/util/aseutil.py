#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ase.io import Trajectory


def downsize_traj(file, l, outfile):
    t = Trajectory(file)
    tt = Trajectory(outfile, 'w')
    for k, atoms in enumerate(t):
        if k == l:
            break
        tt.write(atoms)

