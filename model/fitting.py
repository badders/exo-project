# coding : utf-8
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from scipy.optimize import fmin
from .uniform_disk import occultuniform
from .quad_limb import occultquad
import numpy as np

def fit_uniform_disk(time, flux, flux_err):
    r_p = 0.13
    z = np.linspace(-1.4, 1.4, num=len(time))

    flux = occultuniform(z, r_p)

    return flux, r_p


class ModelFit:
    def __init__(self, res, flux):
        self.res = res
        self.flux = flux

    def model(self, r_p, r_s, stretch, shift):
        z = np.linspace(0, r_s, num=int(self.res*stretch))
        gamma=[0.3, 0.7]
        f2 = occultquad(z, r_p, gamma)
        f1 = f2[::-1]
        flux = np.concatenate((f1, f2))

        if len(flux) < self.res:
            new_flux = np.zeros(self.res)
            new_flux[:len(flux)] = flux
            for i in range(self.res, len(new_flux)):
                new_flux[i] = flux[self.res]
            flux = new_flux

        return flux[:self.res] + shift

    def error(self, params):
        mdata = self.model(*params)
        err = (mdata - self.flux)**2 / mdata
        return err.sum()


def fit_quadlimb(time, flux, flux_err):
    r_p = 0.14
    r_s = 1.4
    stretch = 0.5
    shift = 0
    mf = ModelFit(len(time), flux)
    params = fmin(mf.error, [r_p, r_s, stretch, shift])

    print(params)
    f = mf.model(*params)

    return f, params[0]