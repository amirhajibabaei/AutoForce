import socket
from theforce.util.util import date


class Server:

    def __init__(self, ip, port, callback=None, args=()):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((ip, port))
        self.callback = callback if callback else lambda a: 0
        self.args = args
        with open('server.log', 'w') as log:
            log.write(f'server initiated at: {date()}\n')
            log.write(f'host name: {socket.gethostname()}\n')
            log.write(f'socket name: {self.socket.getsockname()}\n')

    def listen(self, end=b'end', ping=b'?'):
        self.socket.listen(5)
        resume = True
        while resume:
            c, addr = self.socket.accept()
            request = c.recv(1024)
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
