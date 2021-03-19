# +
from theforce.util.statsutil import moving_average
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md import velocitydistribution as vd
from ase.io import read, Trajectory
from ase import units
import numpy as np


def init_velocities(atoms, temperature, overwrite=False, cm0=True, rot0=True):
    """temperature in Kelvin"""
    vel = atoms.get_velocities()
    no_vel = vel is None or np.allclose(vel, 0.)
    if no_vel or overwrite:
        vd.MaxwellBoltzmannDistribution(atoms, temperature*units.kB)
        if cm0:
            vd.Stationary(atoms)
        if rot0:
            vd.ZeroRotation(atoms)


def single_point(self):
    results = {}
    for q in ['energy', 'forces', 'stress']:
        try:
            results[q] = self.calc.results[q]
        except KeyError:
            pass
    self.set_calculator(SinglePointCalculator(self, **results))


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


def get_repeat_reciprocal(atoms, spacing=0.1):
    return np.ceil(np.linalg.norm(
        atoms.get_reciprocal_cell(), axis=1)/spacing).astype(int)


def get_repeat(atoms, spacing=10.):
    return np.ceil(spacing/atoms.cell.cellpar()[:3]).astype(int)


def reciprocal_spacing(atoms):
    return np.linalg.norm(atoms.get_reciprocal_cell(), axis=1)
