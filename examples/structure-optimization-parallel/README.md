### Structure optimization with MPI parallelism

In comparison with the serial version:
* MPI is initialized in the beginning of `relax.py`.
Note that `mpi_init()` also syncs the random number
generator (`numpy.random`) in all processes.
* Ab initio calculators are defined in separate
scripts (`calc_emt.py` or `calc_vasp.py`) and
linked via `SocketCalculator` in `relax.py`.
* The master rank is used for IO.
* A `calc_server` is started before running `relax.py`.
