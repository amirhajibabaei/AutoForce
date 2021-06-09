# +
import numpy as np
from scipy.optimize import curve_fit
from ase.units import kB


def linear_fit(x, y, yerr=None):
    """ 
    returns: [slope, intercept], [slope_sigma, intercept_sigma] 
    """

    def linear(x, m, b):
        return m*x + b

    fit, cov = curve_fit(linear, x, y, sigma=yerr,
                         absolute_sigma=False if yerr is None else True)
    return fit, np.diag(cov)


def arrhenius_fit(T, D, Derr=None):
    """
    fits D = D0 * exp(-Ea/(kB*T))
    return: D0 (same units as D) and Ea (eV) with formats [fit, min, max], [fit, max, min]
    """
    x = 1/(kB*T)
    y = np.log(D)
    yerr = None if Derr is None else np.log(D+Derr) - y
    [m, b], [me, be] = linear_fit(x, y, yerr)
    D0 = np.array([np.exp(b), np.exp(b-be), np.exp(b+be)])
    Ea = np.array([-m, -m+me, -m-me])
    return D0, Ea


def arrhenius_predict(t, D0, Ea):
    return D0*np.exp(-Ea/(kB*t))
