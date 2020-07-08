import os


def forward_port(port, remote, ip='localhost'):
    os.system(f"ssh -N -f -L {ip}:{port}:{ip}:{port} {remote}")


def clear_port(port):
    os.system(f'lsof -ti:{port} | xargs kill -9')
