python -m theforce.calculator.calc_server &
sleep 1
python -m theforce.cl.relax -i Gaussian.gjf -o Gaussian.log

# for MPI parallelism of ML
# mpirun -n 2 python -m theforce.cl.relax -i Gaussian.gjf -o Gaussian.log

# visualization: generates active.pdf
python -m theforce.calculator.active active.log
