# coding : utf-8
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from scipy.optimize import fmin
from .uniform_disk import occultuniform
from .quad_limb import occultquad
import numpy as np
from lmfit import minimize, Parameters, fit_report, report_errors, conf_interval, printfuncs


def fit_uniform_disk(time, flux, flux_err):
    r_p = 0.13
    z = np.linspace(-1.4, 1.4, num=len(time))

    flux = occultuniform(z, r_p)

    return flux, r_p


class ModelFit:
    def __init__(self, res, flux, err):
        self.res = res
        self.flux = flux
        self.err = err

    def model(self, params):
        r_p = params['r_p'].value
        r_s = params['r_s'].value
        stretch = params['stretch'].value
        shift = params['shift'].value

        z = np.linspace(0, r_s, num=int(self.res*stretch))
        gamma=[.2, .8]
        f2 = occultquad(z, r_p, gamma)
        f1 = f2[::-1]
        flux = np.concatenate((f1, f2))

        if len(flux) < self.res:
            new_flux = np.zeros(self.res)
            new_flux[:len(flux)] = flux
            for i in range(len(flux), len(new_flux)):
                new_flux[i] = flux[-1]
            flux = new_flux

        return flux[:self.res] * shift

    def residual(self, params):
        mdata = self.model(params)
        return np.sqrt(((mdata - self.flux)**2 / self.err))


def fit_quadlimb(time, flux, flux_err, stretch=0.5):
    """
    Perform a fit of time/flux data to the quadratic limb darkening model
    """
    params = Parameters()
    params.add('r_p', value=0.12, min=0)
    params.add('r_s', value=1.4, min=0)
    params.add('stretch', value=stretch, min=0, max=1)
    params.add('shift', value=0, min=-0.9, max=1.1)

    mf = ModelFit(len(time), flux, flux_err)
    result = minimize(mf.residual, params)

    f = mf.model(params)
    ci = conf_interval(result, sigmas=[0.95])
    rp_err = ci['r_p'][1][1] - ci['r_p'][0][1]

    print(params['r_p'], rp_err)
    return f, params['r_p'].value, abs(rp_err)