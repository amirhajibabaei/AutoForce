python -m theforce.calculator.calc_server &
sleep 1
mpirun -np 2 python -m theforce.cl.md -i Gaussian.gjf    # <- for MD

### for structure relaxation/optimization
# mpirun -np 2 python -m theforce.cl.relax -i Gaussian.gjf -o Gaussian.log
