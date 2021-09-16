python -m theforce.calculator.calc_server &
sleep 1
mpirun -np 1 python -m theforce.cl.lmp -i in.lammps
