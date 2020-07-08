import os


def forward_port(port, remote):
    os.system(f"ssh -N -f -L localhost:{port}:localhost:{port} {remote}")


def clear_port(port):
    os.system(f'lsof -ti:{port} | xargs kill -9')
