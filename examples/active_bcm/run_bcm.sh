# A. start the calc server
python -m theforce.calculator.calc_server &
sleep 1

# B. start MD
mpirun -np 4 python run_md_bcm.py

# C. stop the calc server (optional)
# you can leave it up for your next MD
echo end | netcat localhost 6666 # stops the calc_server

