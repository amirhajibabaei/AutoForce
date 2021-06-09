# +
from collections import Counter
import itertools
from math import pi
import torch


sq_2pi = torch.tensor(2*pi).sqrt()


def discrete(val, sigma):
    return tuple(val.div(sigma).floor().int().view(-1).tolist())


class Gaussian_kde:

    def __init__(self, sigma, super_grid=5):
        """
        sigma: the band width of the Gaussians. 
        Could be either a scalar or a vector with 
        the same dimensions as the observable.

        super_grid: a positive integer. 
        For efficiency, a super-grid with the width of 
        super_grid*sigma is superimposed. 
        super_grid*sigma is the cutoff for the Gaussians.
        """
        self.sigma = sigma
        self.super_grid = super_grid
        self.data = {}
        self.total = 0

    def __call__(self, x, density=False):
        """
        returns the kde at x.
        if density=True, returns the probability density.
        """
        block = discrete(x, self.super_grid*self.sigma)
        y = 0
        for neihood in itertools.product(*(len(block)*[[-1, 0, 1]])):
            key = tuple((a+b for a, b in zip(block, neihood)))
            if key in self.data:
                counter = self.data[key]
                X = torch.tensor(list(counter.keys())).add(0.5)*self.sigma
                w = torch.tensor(list(counter.values()))
                d2 = (x-X).div(self.sigma).pow(2).sum(dim=-1)
                y = y + d2.mul(-0.5).exp().mul(w).sum()
        if density:
            norm = torch.full_like(x, sq_2pi).mul(self.sigma).prod()*self.total
        else:
            norm = torch.full_like(x, sq_2pi).prod()
        return y/norm

    def count(self, x):
        """
        adds x to the histogram.
        """
        block = discrete(x, self.super_grid*self.sigma)
        if block not in self.data:
            self.data[block] = Counter()
        self.data[block][discrete(x, self.sigma)] += 1.
        self.total += 1

    def histogram(self):
        """
        returns the full histogram.
        """
        points = []
        counts = []
        for block in self.data.values():
            for x, w in block.items():
                points.append(x)
                counts.append(w)
        x = torch.tensor(points).add(0.5)*self.sigma
        w = torch.tensor(counts)
        return x, w


def test_1d():
    import numpy as np
    import pylab as plt

    sigma = 0.1
    kde = Gaussian_kde(sigma)
    data = torch.randn(10000, 1)
    for x in data:
        kde.count(x.view(-1))
    # with numpy
    bins = torch.arange(-5, 5, sigma)
    density, _ = np.histogram(data.numpy(), bins=bins.numpy(), density=True)
    # plot
    X = torch.arange(-5, 5, 0.01)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    # hist
    ax1.scatter(*kde.histogram())
    ax1.plot([], [])  # just to skip the color !
    ax1.plot(X, [kde(x) for x in X], label='kde')
    ax1.set_xlabel('x')
    ax1.set_ylabel('histogram')
    ax1.legend()
    # density
    ax2.plot(bins[:-1]+sigma/2, density, label='numpy')
    ax2.plot(X, [kde(x, True) for x in X], label='kde')
    ax2.set_xlabel('x')
    ax2.set_ylabel('density')
    ax2.legend()
    fig.tight_layout()


def test_2d():
    import numpy as np
    import pylab as plt

    sigma = 0.25
    kde = Gaussian_kde(sigma)
    data = torch.randn(100000, 2)
    for x in data:
        kde.count(x.view(-1))

    errors = []
    for _ in range(1000):
        x = torch.rand(2)
        a = kde(x, density=True)
        b = x.pow(2).sum().mul(-0.5).exp()/sq_2pi**2
        errors.append(float(a-b))

    _ = plt.hist(errors, bins=20)
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('error')
    ax.hist(errors)
    ax.set_ylabel('histogram')
    fig.tight_layout()


if __name__ == '__main__':
    test_1d()
    test_2d()
