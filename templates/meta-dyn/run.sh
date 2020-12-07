python -m theforce.calculator.calc_server &
sleep 1
mpirun -np 6 python md.py
echo end | netcat localhost 6666 
