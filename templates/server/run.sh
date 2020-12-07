python -m theforce.calculator.calc_server &
sleep 1
python socalc.py
echo end | netcat localhost 6666 
