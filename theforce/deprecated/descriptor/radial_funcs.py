import numpy as np


class constant:
    def __init__(self, rc):
        """rc"""
        self.rc = rc

    def radial(self, r):
        return self.rc, 0


class gaussian:
    def __init__(self, rc):
        """exp( - r^2 / 2 rc^2 )"""
        self.rc = rc

    def radial(self, r):
        x = -r / self.rc**2
        y = np.exp(x * r / 2)
        return y, x * y


class cosine_cutoff:
    def __init__(self, rc):
        """( 1 + cos( pi * r / rc ) ) / 2"""
        self.rc = rc

    def radial(self, r):
        x = np.pi * r / self.rc
        y = (1 + np.cos(x)) / 2
        outlier = r >= self.rc
        y[outlier] = 0.0
        dy = -np.pi * np.sin(x) / (2 * self.rc)
        dy[outlier] = 0.0
        return y, dy


class quadratic_cutoff:
    def __init__(self, rc):
        """( 1 - r / r_c )^2"""
        self.rc = rc

    def radial(self, r):
        x = 1.0 - r / self.rc
        x[x < 0.0] = 0.0
        return x * x, -2 * x / self.rc


class poly_cutoff:
    def __init__(self, rc, n):
        """( 1 - r / r_c )^n"""
        self.rc = rc
        self.n = n
        self.n_ = n - 1

    def radial(self, r):
        x = 1.0 - r / self.rc
        x[x < 0.0] = 0.0
        y = x ** (self.n_)
        return x * y, -self.n * y / self.rc
