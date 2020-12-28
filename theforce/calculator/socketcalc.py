# +
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from theforce.util.util import date
import torch
import socket
import os


class SocketCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, ip='localhost', port=6666, script=None, wlog=False, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.ip = ip
        self.port = port
        self.script = script
        self.wlog = wlog
        self.log('created', 'w')

    def log(self, msg, mode='a'):
        if self.rank == 0 and self.wlog:
            with open('socalc.log', mode) as f:
                f.write(f'{date()}   {msg}\n')

    def ping(self):
        s = socket.socket()
        s.connect((self.ip, self.port))
        s.send(b'?')
        print(f'server says: {s.recv(1024)}')
        s.close()

    @property
    def is_distributed(self):
        return torch.distributed.is_initialized()

    @property
    def rank(self):
        if self.is_distributed:
            return torch.distributed.get_rank()
        else:
            return 0

    @property
    def message(self):
        cwd = os.getcwd()
        msg = f'{cwd}/socket_send.xyz:{cwd}/socket_recv.xyz'
        if self.script is not None:
            msg = f'{msg}:{os.path.abspath(self.script)}'
        return msg

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # write and send request
        self.log('s')
        ierr = 0
        if self.rank == 0:
            s = socket.socket()
            s.connect((self.ip, self.port))
            self.atoms.write('socket_send.xyz')
            s.send(self.message.encode())
            ierr = int(s.recv(1024).decode('utf-8'))
            s.close()
        assert ierr == 0
        if self.is_distributed:
            torch.distributed.barrier()
        self.log('e')
        # read
        atms = read('socket_recv.xyz')
        self.results['energy'] = atms.get_potential_energy()
        self.results['forces'] = atms.get_forces()
        self.results['stress'] = atms.get_stress()
        # delete files
        if self.is_distributed:
            torch.distributed.barrier()
        if self.rank == 0:
            os.system('rm -f socket_send.xyz socket_recv.xyz')

    def close(self):
        s = socket.socket()
        s.connect((self.ip, self.port))
        s.send(b'end')
        s.close()
