# +
import argparse
import os
import socket
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="starts twin processes")
    parser.add_argument("pyscript")
    parser.add_argument("-ip", "--ip", default="localhost")
    parser.add_argument("-port", "--port", type=int, default=6666)
    parser.add_argument("-np", "--mpiprocs", default=None)
    args, unknown = parser.parse_known_args()

    # calc command
    calc = f"python -m theforce.calculator.calc_server -ip {args.ip} -port {args.port}"

    # mpirun pyscript command
    if args.mpiprocs is not None:
        mpiargs = f"-np {args.mpiprocs}"
    else:
        mpiargs = ""
    extra = " ".join(unknown)
    mpi = f"mpirun {mpiargs} python {args.pyscript} {extra}"

    # run
    command = f"{calc} & {mpi}"
    print(f"running: {command}")
    assert os.system(command) == 0

    # close calc socket
    try:
        s = socket.socket()
        s.connect((args.ip, args.port))
        s.send(b"end")
        s.close()
    except:
        pass
