python gen_images.py


python -m theforce.calculator.calc_server &
sleep 1
mpirun -n 6 python -m theforce.cl.neb -i images.traj

# recommended: run a second time with a more accurate ML model
mpirun -n 6 python -m theforce.cl.neb -i images.traj


python visualize.py
