python -m theforce.calculator.calc_server &
sleep 1
mpirun -n 4 python md.py
