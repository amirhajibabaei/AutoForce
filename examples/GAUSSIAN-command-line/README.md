<!-- #region -->
### On-the-fly MLMD with GAUSSIAN
* make sure that one of g16, g09, or g03 is in the PATH (alternatively the `COMMAND` file can be used)
* create the input Gaussian job file (Gaussian.gjf)
* set the job-type for single-point energy and force calculations
* create `ARGS` file for setting ML and MD params
* use `./run.sh` for running the example
* the log can be visualized as follows
```sh
python -m theforce.calculator.active active.log
```

An example is given here.
It is assumed that 6 cores are available which are split as 2 for ML, 4 for GAUSSIAN.
You can modify the cores for GAUSSIAN in `Gaussian.gjf` and for ML in `run.sh`.

#### Structure relaxtion/optimization
* notes for relaxation (instead of MD) are given as comments in `ARGS` and `run.sh`
<!-- #endregion -->
