import socket


class Server:

    def __init__(self, ip, port, callback=None, args=()):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((ip, port))
        self.callback = callback if callback else lambda a: 0
        self.args = args

    def listen(self, end=b'end'):
        self.socket.listen(5)
        resume = True
        while resume:
            c, addr = self.socket.accept()
            request = c.recv(1024)
            if request == end:
                resume = False
            else:
                self.callback(request, *self.args)
                c.send(b'0')
            c.close()
        self.socket.close()
