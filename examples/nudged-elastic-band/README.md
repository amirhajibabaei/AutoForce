<!-- #region -->
### Nudged-Elastic-Band with on-the-fly ML

This example is copied and modified from:
<https://wiki.fysik.dtu.dk/ase/tutorials/neb/diffusion.html>.

`gen_images.py` generates `images.traj` for starting NEB.
The commands for running are available in `run.sh`.
The optimized band is written to `neb.traj`.
`visualize.py` generated the barrier figure.

Tight ML parameters (`ediff` and `fdiff`) are chosen
for accuracy.
In practice one can choose larger thresholds for
faster calculations in exchange for slightly larger
errors (~0.1 eV).

Notice that relaxation of the first and last images
is integrated with NEB and one is not required
to provide optimized structures.

`climb = True` directs the program to perform
a climbing image NEB.

In this example, the entire band is provided
as input.
Alternatively one can input only the first
and last images via:
```bash
mpirun -n 6 python -m theforce.cl.neb -i first 5 last -o opt-band.traj
```
where first or last can be any ASE readable format
(traj, xyz, POSCAR, etc).
"5" directs the algorithm to create 5 images between
the first and the last one.
<!-- #endregion -->
