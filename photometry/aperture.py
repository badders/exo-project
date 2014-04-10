# coding : utf-8
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from collections import namedtuple
import math

from astropy.io import fits
from common.gaussian import fitgaussian2D


Aperture = namedtuple('Aperture', 'x y r br1 br2')
AP_MIN = 5
AP_MIN_B1 = 6
AP_MIN_B2 = 8

def filter_overlapping(apertures):
    return apertures


def filter_saturated(im, apertures, threshold=0.8):
    max_value = threshold * fits.open(im)[0].header['DATAMAX']
    return apertures


def generate_apertures(im, sources, max_radius=30):
    apertures = []
    data = fits.open(im)[0].data
    for s in sources:
        y_min = math.floor(max(s.y - max_radius, 0))
        y_max = math.floor(min(s.y + max_radius, data.shape[0] - 1))
        x_min = math.ceil(max(s.x - max_radius, 0))
        x_max = math.ceil(min(s.x + max_radius, data.shape[1] - 1))

        if y_min < y_max - 1 and x_min < x_max - 1:
            star = data[y_min:y_max, x_min:x_max]
            params = fitgaussian2D(star)
            r = max(params[3], params[4])
            if 2 * r <= max_radius and 2*r > 0:
                apertures.append(Aperture(s.x, s.y, 3 * r, 3*r + r, 3*r + 2*r))
            else:
                apertures.append(Aperture(s.x, s.y, AP_MIN, AP_MIN_B1, AP_MIN_B2))


    return apertures