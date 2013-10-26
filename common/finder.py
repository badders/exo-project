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
from gaussian import gaussian2D,  fitgaussian2D

DEBUG = False

# Default Parameters
SNR = 5
BG_THRESHOLD_PC = 30.
SEARCH_RADIUS = 10


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
        y1, y2 = math.floor(y - radius), math.floor(y + radius)
        x1, x2 = math.floor(x - radius), math.floor(x + radius)
        star = data[y1:y2, x1:x2]
        params = fitgaussian2D(star)
        ny, nx = params[1:3]
        centered_stars[i] = np.array([x1 + nx, y1 + ny])

        if i == 56 and DEBUG:
            plt.imshow(star)
            plt.contour(gaussian2D(*params)(*np.indices(star.shape)))
            plt.hlines([ny], *plt.ylim())
            plt.vlines([nx], *plt.xlim())
            plt.tight_layout()

    return stars, centered_stars

if __name__ == '__main__':
    DEBUG = True
    test_image = '/Users/tom/fits/transition/qatar1b-1.fits'
    stars, centered = find_star_coords(test_image, snr=3.5)
    fig = aplpy.FITSFigure(test_image)
    fig.show_colorscale(cmap='gist_heat')
    plt.plot(stars[:, 0], stars[:, 1], 'y+', markersize=8)
    plt.plot(centered[:, 0], centered[:, 1], 'g+', markersize=8)

    counter = 0
    for star in stars:
        plt.annotate(counter, xy=(star[0], star[1]), xytext=(-10, 10),
                     textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='y', alpha=0.2),
                     arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0'))
        counter += 1
    plt.tight_layout()
    plt.show()
