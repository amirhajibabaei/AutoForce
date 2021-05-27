### On-the-fly MLMD with VASP
* setup the VASP calculator as explained in ASE docs
* create the VASP input files as usual (only for single-point calculatios: IBRION=-1, NSW=0)
* create `ARGS` file for setting ML and MD params
* use `./run.sh` for running the example
* the log can be visualized as follows
```sh
python -m theforce.calculator.active active.log
```

An example is given here.
It is assumed that 20 cores are available which are split as 8 for ML, 12 for VASP.
You can modify the cores for VASP in `COMMAND` and for ML in `run.sh`.
