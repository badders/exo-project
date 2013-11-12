from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from photometry import finder
import glob
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from os import path
from common.display import show_fits

FLATS = '/Users/tombadran/fits/test-data/flats/*.FIT'
BIAS = '/Users/tombadran/fits/test-data/bias/*.FIT'
IMAGES = '/Users/tombadran/fits/test-data/raw/*.FIT'
CORRECTED_DEST = '/Users/tombadran/fits/test-data/corrected/'

DEBUG_IMAGE = '/Users/tombadran/fits/test-data/raw/hat-p-8-b3001.FIT'
DEBUG_FLAT = '/Users/tombadran/fits/test-data/flats/flata001.FIT'
DEBUG_BIAS = '/Users/tombadran/fits/test-data/bias/test001.FIT'


def generate_bias(pathname):
    """
    Load the images from pathname and generate a bias correction
    """
    images = glob.glob(pathname)
    fits_images = [fits.open(f) for f in images]
    bias = np.zeros(fits_images[0][0].shape, dtype=np.float64)

    for im in fits_images:
        bias += im[0].data

    bias = bias / len(fits_images)

    return bias


def generate_flat(pathname):
    """
    Load the images in the given folder, and generate a flat field
    correction from the data
    """
    images = glob.glob(pathname)
    fits_images = [fits.open(f) for f in images]
    flat = np.zeros(fits_images[0][0].shape, dtype=np.float64)

    for im in fits_images:
        flat += im[0].data

    flat = flat / flat.mean()
    return flat


def correct_images(pathname, dark_frame=None, flat_frame=None):
    """
    Load a set of images and correct them using optional dark and flat frames
    """
    images = glob.glob(pathname)
    fits_images = [(path.basename(f), fits.open(f)) for f in images]

    corrected_images = []
    for f, hdulist in fits_images:
        print('Correcting {}'.format(f))

        if dark_frame is not None:
            hdulist[0].data = hdulist[0].data - dark_frame

        if flat_frame is not None:
            hdulist[0].data = hdulist[0].data / flat_frame
        fn = CORRECTED_DEST + f
        hdulist.writeto(fn, clobber=True)
        corrected_images.append(fn)
    return corrected_images


if __name__ == '__main__':
    #finder.test(DEBUG_IMAGE, snr=3)
    bias = generate_bias(DEBUG_BIAS)
    flat = generate_flat(DEBUG_FLAT)
    #show_fits(DEBUG_IMAGE)
    #show_fits(flat)
    im = correct_images(DEBUG_IMAGE, dark_frame=bias, flat_frame=flat)
    #show_fits(im[0])
    finder.test(im[0], snr=5)
    plt.show()
