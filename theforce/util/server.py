import socket
from theforce.util.util import date


class Server:

    def __init__(self, ip, port, callback=None, args=(), wlog=False):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((ip, port))
        self.callback = callback if callback else lambda a: 0
        self.args = args
        h = socket.gethostname()
        s = self.socket.getsockname()
        self.wlog = wlog
        self.log(f'server initiated at: {h} {s}', 'w')

    def log(self, msg, mode='a'):
        if self.wlog:
            with open('server.log', mode) as log:
                log.write(f'{date()}: {msg}\n')
        else:
            print(f'{date()}: {msg}\n')

    def listen(self, end='end', ping='?'):
        self.socket.listen(5)
        resume = True
        while resume:
            c, addr = self.socket.accept()
            request = c.recv(1024).decode("utf-8").strip()
            self.log(request)
            if request == end:
                resume = False
            elif request == ping:
                c.send(b'!')
            else:
                try:
                    self.callback(request, *self.args)
                    c.send(b'0')
                except:
                    c.send(b'-1')
            c.close()
        self.socket.close()
