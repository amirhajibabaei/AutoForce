# generate atoms.xyz
python make_atoms.py

# start the calc server
python -m theforce.calculator.calc_server &
sleep 1

# start MD
mpirun -np 6 python -m theforce.cl.md -i atoms.xyz

