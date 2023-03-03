# +
import os
import time


def forward_port(port, remote, ip="localhost"):
    os.system(f"ssh -N -f -L {ip}:{port}:{ip}:{port} {remote}")


def clear_port(port):
    # os.system(f'lsof -ti:{port} | xargs kill -9')
    procs = os.popen(f"lsof -ti:{port}").read().split()
    for p in procs:
        print(f"\nkilling process {p} occupying port {port}")
        os.system(f"kill -9 {p}")
        time.sleep(0.1)
        print("killed!")
