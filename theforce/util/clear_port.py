# +
import os


def clear_port(port):
    return [(process, os.system(f'kill -9 {process}')) for process in
            os.popen(f'lsof -ti:{port}').read().split()]


if __name__ == '__main__':
    import sys
    print(f'killed processes: {clear_port(sys.argv[1])}')
