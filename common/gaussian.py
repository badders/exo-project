# coding : utf-8
"""
Gaussian and fitting functions
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)


import numpy as np
from scipy import optimize


def gaussian2D(height, cx, cy, w, h, c):
    """Returns a gaussian function with the given parameters"""
    return lambda x, y: height * np.exp(-(((cx - x) / w)**2 + ((cy - y) / h)**2) / 2) + c


def fitgaussian2D(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    def errorfunction(p):
        return np.ravel(gaussian2D(*p)(*np.indices(data.shape)) - data)
    max_pos = np.unravel_index(data.argmax(), data.shape)
    params = data.max(), max_pos[0], max_pos[1], 10, 10, np.median(data)
    p, success = optimize.leastsq(errorfunction, params)
    return p
