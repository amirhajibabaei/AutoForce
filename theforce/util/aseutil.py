from theforce.util.statsutil import moving_average
from ase.io import read
import numpy as np
from ase.io import Trajectory


def downsize_traj(file, l, outfile):
    t = Trajectory(file)
    tt = Trajectory(outfile, 'w')
    for k, atoms in enumerate(t):
        if k == l:
            break
        tt.write(atoms)


def write_average_positions(traj, w, sl='::', out=None):
    refatoms = read(traj, 0)
    positions = moving_average(
        np.stack([atoms.positions for atoms in read(traj, sl)]), w)
    cell = moving_average(
        np.stack([atoms.cell for atoms in read(traj, sl)]), w)
    if out is None:
        if traj.endswith('.traj'):
            out = traj.replace('.traj', '_ave.traj')
        else:
            raise RuntimeError('file does not end with .traj')
    t = Trajectory(out, 'w')
    for p, c in zip(*[positions, cell]):
        refatoms.set_positions(p)
        refatoms.set_cell(c)
        t.write(refatoms)
    print(f'average positions are written to {out}')


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

