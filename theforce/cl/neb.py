# +
from ase import optimize
from ase.io import Trajectory, read
from ase.neb import NEB

import theforce.cl as cline
from theforce.cl.relax import relax


def nudged_elastic_band(
    images,
    fmax=0.01,
    climb=False,
    algo="BFGS",
    rel_if=1,
    algo_if="BFGS",
    trajectory="neb-path.traj",
    output="neb-out.traj",
):
    """
    images:       list of atoms objects
    fmax:         maximum forces
    climb:        for climbing image NEB
    algo:         optim algo (from ase.optimize) for NEB
    rel_if:       relax initial and final images: 0 (no), 1 (once), 2 (every)
    algo_if:      optim algo (from ase.optimize) for first/last images
    trajectory:   traj file name for neb path
    output:       traj file name for neb optimized band
    """

    calc = cline.gen_active_calc()
    load1 = calc.size[0]
    master = calc.rank == 0

    for image in images:
        image.calc = calc
        image.get_potential_energy()

    def relax_if(confirm=False):
        suff = {0: "first", -1: "last"}
        for i in [0, -1]:
            label = suff[i]
            if master:
                print(f"Relaxing the {label} image  ... ")
            relax(
                images[i],
                fmax=fmax,
                calc=calc,
                rattle=0.0,
                confirm=confirm,
                algo=algo_if,
                trajectory=f"relax_{label}.traj",
            )

    def relax_is_allowed():
        if rel_if == 0:
            return False
        elif rel_if == 1:
            return not resuming
        else:
            return True

    # define and interpolate
    def converge():
        if relax_is_allowed():
            relax_if(confirm=not resuming)
        neb = NEB(images, climb=climb, allow_shared_calculator=True)
        Min = getattr(optimize, algo)
        dyn = Min(neb, trajectory=trajectory, master=master)
        size1 = calc.size[0]
        if master:
            print("(Re)starting NEB ...")
        for _ in dyn.irun(fmax):
            if calc.size[0] > size1:
                if master:
                    print("model updated -> restart!")
                return False
        return True

    resuming = False
    while not converge():
        resuming = True

    load2 = calc.size[0]
    if master:
        print("\tNEB finished!")
        print(f"\tTotal number of Ab initio calculations: {load2-load1}\n")
        out = Trajectory(output, "w")

    for image in images:
        image.get_potential_energy()
        if master:
            out.write(image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Machine Learning accelerated NEB. "
        "It will also try to relax the initial and final images "
        "thus there is no need for prior relaxation."
        "For setting constraints, use ASE and save the images "
        "in a .traj file."
    )
    parser.add_argument(
        "-i",
        "--images",
        default="images.traj",
        type=str,
        nargs="*",
        help="List of files for reading the images: "
        "-i images.traj (default) "
        "OR -i file1 7 file2 which generates 7 images in between "
        "OR explicitly -i file1 file2 file3 ... "
        "WHERE the input files can be .traj, .xyz, POSCAR, etc.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="neb-out.traj",
        type=str,
        help="Filename for writing the optimized band (default: neb-out.traj)",
    )
    args = parser.parse_args()

    # read
    images = []
    interpolate = False
    for i in args.images:
        if i.isdecimal():
            images.extend([images[-1].copy() for _ in range(int(i))])
            interpolate = True
        else:
            images.extend(read(i, ":"))
    if interpolate:
        NEB(images).interpolate()

    # get other ARGS
    kwargs = cline.get_default_args(nudged_elastic_band)
    cline.update_args(kwargs)
    kwargs["output"] = args.output

    # run
    nudged_elastic_band(images, **kwargs)
