# +
import os

import numpy as np
from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.npt import NPT

import theforce.cl as cline
from theforce.calculator.active import FilterDeltas
from theforce.util.aseutil import init_velocities, make_cell_upper_triangular


def md(
    atoms,
    dynamics="NPT",
    dt=None,
    tem=300.0,
    picos=100,
    bulk_modulus=None,
    stress=0.0,
    mask=None,
    iso=False,
    trajectory="md.traj",
    loginterval=1,
    append=False,
    rattle=0.0,
    tdamp=25,
    pdamp=100,
    friction=1e-3,
    ml_filter=0.8,
    eps_pos=0.05,
    eps_cell=0.05,
):
    """
    atoms:        ASE atoms
    dynamics:     'NPT'
    dt:           time-step in fs
    tem:          temperature in Kelvin
    picos:        pico-seconds for md
    bulk_modulus: bulk_modulus for NPT simulations. if None, NVT is performed
    stress:       external stress (GPa) for NPT
    mask:         see ase.npt.NPT
    iso:          if True, keep the shape constant
    trajectory:   traj file name
    loginterval:  for traj file
    append:       append to traj file
    rattle:       rattle atoms at initial step (recommended ~0.05)
    tdamp:        temperature damping time (fs)
    pdamp:        pressure damping time (fs)
    friction:     for Langevin dynamics
    ml_filter:    filters force discontinuities due to ML updates range(0, 1)
    """

    calc = cline.gen_active_calc()
    atoms.calc = calc
    if calc.active:
        manual_steps(atoms, eps_pos, eps_cell, npt=bulk_modulus)
    atoms.rattle(rattle, rng=np.random)

    Ts = get_temperatures(tem)
    if calc.rank == 0:
        print(f"MD temperatures: {Ts}")
    init_velocities(atoms, Ts[0])
    atoms.get_potential_energy()
    if calc.deltas:
        calc.results.clear()

    if dt is None:
        if (atoms.numbers == 1).any():
            dt = 0.25
        else:
            dt = 1.0

    if ml_filter:
        md_atoms = FilterDeltas(atoms, shrink=ml_filter)
    else:
        md_atoms = atoms

    for T in Ts:
        if dynamics.upper() == "NPT":
            dyn = npt_dynamics(
                md_atoms,
                dt,
                T,
                bulk_modulus,
                stress,
                mask,
                iso,
                trajectory,
                loginterval,
                append,
                tdamp,
                pdamp,
            )
        elif dynamics.upper() == "LANGEVIN":
            dyn = langevin_dynamics(
                md_atoms, dt, T, friction, trajectory, loginterval, append
            )

        if calc.meta is not None:
            dyn.attach(calc.meta.update)

        steps = int(picos * 1000 / dt) if picos > 0 else -picos
        dyn.run(steps)
        append = True


def get_temperatures(tem):
    if hasattr(tem, "__iter__"):
        return tem
    else:
        return [tem]


def langevin_dynamics(atoms, dt, tem, friction, trajectory, loginterval, append):
    dyn = Langevin(
        atoms,
        dt * units.fs,
        temperature_K=tem,
        friction=friction,
        rng=np.random,
        trajectory=trajectory,
        append_trajectory=append,
        loginterval=loginterval,
    )
    return dyn


def npt_dynamics(
    atoms,
    dt,
    tem,
    bulk_modulus,
    stress,
    mask,
    iso,
    trajectory,
    loginterval,
    append,
    tdamp,
    pdamp,
):
    ttime = tdamp * units.fs
    ptime = pdamp * units.fs
    if bulk_modulus:
        pfactor = (ptime**2) * bulk_modulus * units.GPa
    else:
        pfactor = None
    configure_cell(atoms)
    dyn = NPT(
        atoms,
        dt * units.fs,
        temperature_K=tem,
        externalstress=stress * units.GPa,
        ttime=ttime,
        pfactor=pfactor,
        mask=mask,
        trajectory=trajectory,
        append_trajectory=append,
        loginterval=loginterval,
    )
    if iso:
        dyn.set_fraction_traceless(0.0)
    return dyn


def configure_cell(atoms):
    if np.allclose(atoms.cell, 0.0):
        atoms.center(vacuum=6.0)
    make_cell_upper_triangular(atoms)


def manual_steps(atoms, eps, eps2, npt=False):
    calc = atoms.calc
    calc._logpref = "#"
    calc.log("manual steps:")
    calc.log(f"rattle: {eps}")
    positions = atoms.positions.copy()
    if eps > 0.0:
        atoms.rattle(eps, rng=np.random)
        atoms.get_potential_energy()
    if npt and eps2 > 0.0:
        cell = atoms.cell.copy()
        calc.log(f"expand: {(1.+eps2)}*cell")
        atoms.set_cell((1.0 + eps2) * cell, scale_atoms=True)
        atoms.get_potential_energy()
        calc.log(f"shrink: {(1.-eps2)}*cell")
        atoms.set_cell((1.0 - eps2) * cell, scale_atoms=True)
        atoms.get_potential_energy()
        atoms.set_cell(cell, scale_atoms=True)
    atoms.positions = positions
    calc._logpref = ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Machine Learning Molecular Dynamics (MLMD)"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="POSCAR",
        type=str,
        help="the initial coordinates of atoms, POSCAR, xyz, cif, etc.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="CONTCAR",
        type=str,
        help="the final coordinates of atoms",
    )
    args = parser.parse_args()
    if args.input.endswith(".traj"):
        atoms = read(args.input, -1)
    else:
        atoms = read(args.input)
    kwargs = cline.get_default_args(md)
    cline.update_args(kwargs)
    md(atoms, **kwargs)
    try:
        atoms.write(args.output)
    except:
        import warnings

        alt = "md.final.xyz"
        msg = f"writing to {args.output} failed -> wrote {alt}"
        warnings.warn(msg)
        atoms.write(alt)
