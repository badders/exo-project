# coding : utf-8
"""
Finds stars on a fits image
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from astropy.io import fits
import aplpy
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage.filters as filters
from scipy.ndimage import label, find_objects
from scipy import optimize
import math


# Default Parameters
SNR = 5
BG_THRESHOLD_PC = 30.
SEARCH_RADIUS = 10


def gaussian(height, cx, cy, w, h, c):
    """Returns a gaussian function with the given parameters"""
    return lambda x, y: height * np.exp(-(((cx - x) / w)**2 + ((cy - y) / h)**2) / 2) + c


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    def errorfunction(p):
        return np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    params = data.max(), data.shape[0] / 2, data.shape[1] / 2, 5, 5, data.mean()
    p, success = optimize.leastsq(errorfunction, params)
    return p


def find_star_coords(image_file, snr=SNR, radius=SEARCH_RADIUS):
    """
    Find all the stars in a fits file, and return their x/y coordinates in the
    data.
    """
    hdulist = fits.open(image_file)
    data = hdulist[0].data

    # 1. Calculate the average bavckground value
    lower_bound = np.percentile(data, BG_THRESHOLD_PC)
    bg = (data[data < lower_bound]).mean()

    # 2. Find local maxima above our SNR
    data_max = filters.maximum_filter(data, radius)
    maxima = (data == data_max)
    diff = (data_max >= bg * snr)
    maxima[diff == 0] = 0

    height, width = data.shape

    labeled = label(maxima)[0]
    slices = find_objects(labeled)
    stars = []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        y_center = (dy.start + dy.stop - 1) / 2

        valid = True
        if x_center < radius or y_center < radius:
            valid = False
        if x_center > width - radius or y_center > width - radius:
            valid = False
        if valid:
            stars.append([x_center, y_center])
    stars = np.array(stars)
    centered_stars = np.zeros_like(stars)

    # 3. Now refine these maxima to star central points
    for i in range(stars.shape[0]):
        x, y = stars[i]
        y1, y2 = math.floor(y - radius), math.ceil(y + radius)
        x1, x2 = math.floor(x - radius), math.ceil(x + radius)
        star = data[y1:y2, x1:x2]
        params = fitgaussian(star)
        ny, nx = params[1:3]
        centered_stars[i] = np.array([x1 + nx, y1 + ny])

        if i == 40:
            plt.imshow(star)
            plt.contour(gaussian(*params)(*np.indices(star.shape)))
            plt.hlines([ny], *plt.ylim())
            plt.vlines([nx], *plt.xlim())
            plt.tight_layout()

    return centered_stars

if __name__ == '__main__':
    test_image = '/Users/tom/fits/transition/qatar1b-1.fits'
    stars = find_star_coords(test_image, snr=3.5)
    fig = aplpy.FITSFigure(test_image)
    fig.show_colorscale(cmap='hot')
    plt.plot(stars[:, 0], stars[:, 1], 'r+', markersize=8)
    plt.tight_layout()
    plt.show()
