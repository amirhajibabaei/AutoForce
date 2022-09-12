# A. start the calc server
python -m theforce.calculator.calc_server &
sleep 1

# B. start MD
mpirun -np 20 python relax.py

# C. stop the calc server (optional)
# you can leave it up for your next MD
echo end | netcat localhost 6666 # stops the calc_server

# Notes:
# 1. the default ip and port are 
#    ip: localhost
#    port: 6666
#    for changing these, pass them as args when
#    starting the server in step A and in defining
#    the SocketCalculator in md.py step B
#
# 2. Note that A and B can be executed on different 
#    nodes/machines. Just use ssh forwarding to connect 
#    ip and ports


# visualization: generates "active.pdf"
python -m theforce.calculator.active active.log
