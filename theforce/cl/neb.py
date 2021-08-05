# +
import theforce.cl as cline
from ase.neb import NEB
from ase import optimize
from ase.io import read, Trajectory


def nudged_elastic_band(images, fmax=0.01, algo='BFGS', trajectory='neb-path.traj'):

    calc = cline.gen_active_calc()
    load1 = calc.size[0]
    master = calc.rank == 0

    for image in images:
        image.calc = calc

    # calculate for the first and last images
    # (for more efficient ML)
    images[0].get_potential_energy()
    images[-1].get_potential_energy()

    # define and run NEB
    neb = NEB(images, allow_shared_calculator=True)
    Min = getattr(optimize, algo)
    dyn = Min(neb, trajectory=trajectory, master=master)
    dyn.run(fmax)

    load2 = calc.size[0]
    if master:
        print(f'\tTotal number of Ab initio calculations: {load2-load1}\n')

    # output
    if master:
        out = Trajectory('neb.traj', 'w')
    for image in images:
        image.get_potential_energy()
        if master:
            out.write(image)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Machine Learning accelerated NEB')
    parser.add_argument('-i', '--images', default='images.traj', type=str,
                        help='initial images for NEB (ASE traj file)')
    args = parser.parse_args()
    images = read(args.images, ':')
    kwargs = cline.get_default_args(nudged_elastic_band)
    cline.update_args(kwargs)
    nudged_elastic_band(images, **kwargs)
