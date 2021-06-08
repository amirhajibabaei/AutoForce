### Structure optimization
In this example a random cluster of 20 gold atoms
is generated and relaxed until ML forces are less
than 0.01.
Due to the ML errors the actual ab initio forces
are slightly larger (~0.05) but the ML-optimized
cluster should be very close to a local minima.
For testing we used `EMT` instead of `VASP`.
Note that in consecutive executions:
* Every time a new random cluster is generated.
* The ML model is automatically saved and reused
(via `model.pckl/`).

Therefore as the model matures, less ab initio
calculations are invoked.
Nevertheless, in each optimization at least
two ab initio calculations are needed:
* Every time the structure is optimized
(according to the ML potential), for finer
optimization, an ML update+reoptimization is
tried (at least 1 ab initio calculation).
* At the end, 1 ab initio calculation is
carried out for obtaining  the exact
residual forces.
