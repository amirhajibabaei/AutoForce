python -m theforce.calculator.calc_server &
sleep 1
mpirun -np 4 python md.py
