python gen_images.py


python -m theforce.calculator.calc_server &
sleep 1
mpirun -n 6 python -m theforce.cl.neb -i images.traj -o opt-band.traj


python visualize.py
