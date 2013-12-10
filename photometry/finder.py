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
from scipy.stats import mode
from scipy import optimize
from common.dependency import update_required
import math
from common.gaussian import gaussian2D,  fitgaussian2D
from common.display import show_fits
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.time import Time
from os import path
import subprocess

DEBUG = False
DEBUG_STAR = 24

# Default Parameters
SNR = 3
BG_THRESHOLD_PC = 95.
SEARCH_RADIUS = 10
SNR_FLUX_RADIUS = 4

def check_around_point(data, bg, snr, x_center, y_center):
    for x in range(-SNR_FLUX_RADIUS, SNR_FLUX_RADIUS):
        for y in range(-SNR_FLUX_RADIUS, SNR_FLUX_RADIUS):
            if x**2 + y**2 < SNR_FLUX_RADIUS**2:
                pass

    return True

def find_star_coords(image_file, snr=SNR, radius=SEARCH_RADIUS):
    """
    Find all the stars in a fits file, and return their x/y coordinates in the
    data.
    """
    hdulist = fits.open(image_file)
    data = hdulist[0].data

    # 1. Remove anything lower than our SNR
    #bg = mode(data)

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
        valid = valid and check_around_point(data, bg, snr, x_center, y_center)
        if valid:
            stars.append([x_center, y_center])
    stars = np.array(stars)
    centered_stars = []

    # 3. Now refine these maxima to star central points
    for i in range(stars.shape[0]):
        x, y = stars[i]
        y1, y2 = math.floor(y - radius), math.floor(y + radius) + 1
        x1, x2 = math.floor(x - radius), math.floor(x + radius) + 1
        star = data[y1:y2, x1:x2]
        params = fitgaussian2D(star)
        ny, nx = params[1:3]
        centered_stars[i] = np.array([x1 + nx, y1 + ny])

        if i == DEBUG_STAR and DEBUG:
            print(x1, x2, y1, y2, nx, ny)
            plt.imshow(star)
            plt.contour(gaussian2D(*params)(*np.indices(star.shape)))
            plt.hlines([ny], *plt.ylim())
            plt.vlines([nx], *plt.xlim())
            plt.tight_layout()

    return stars, centered_stars


def run_sextractor(images, force=False, DATA_DEST='', **params):
    param_str = ''
    for p, v in params:
        param_str += '-{} {} '.format(p, v)

    output = []

    for i in range(0, len(images)):
        RESULTS_FILE = DATA_DEST + path.basename(images[i]) + '.dat'
        if force or update_required(RESULTS_FILE, images[i]) or update_required(RESULTS_FILE, 'default.param'):
            subprocess.check_output('/opt/local/bin/sex {} -c config.sex '.format(images[i]) +
                                    param_str,
                                    shell=True)
            se = ascii.SExtractor()
            data = se.read(open('test.cat').read())
            data.write(RESULTS_FILE, format='ascii')
        else:
            data = Table.read(RESULTS_FILE, format='ascii')

        time = Time(fits.open(images[i])[0].header['DATE-OBS'], scale='utc')

        output.append((time, data))

    return output

def test(test_image='/Users/tombadran/fits/transition/qatar1b-1.fits', snr=3, radius=SEARCH_RADIUS):
    global DEBUG
    DEBUG = True
    stars, centered = find_star_coords(test_image, snr=snr)
    show_fits(test_image)
    plt.plot(stars[:, 0], stars[:, 1], 'y+', markersize=8)
    plt.plot(centered[:, 0], centered[:, 1], 'g+', markersize=8)

    counter = 0
    for star in centered:
        plt.annotate(counter, xy=(star[0], star[1]), xytext=(-10, 10),
                     textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='y', alpha=0.2),
                     arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0'))
        counter += 1
    plt.show()


if __name__ == '__main__':
    test()
