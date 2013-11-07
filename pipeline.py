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
IMAGES = '/Users/tombadran/fits/test-data/raw/*.FIT'
CORRECTED_DEST = '/Users/tombadran/fits/test-data/corrected/'
DEBUG_IMAGE = '/Users/tombadran/fits/test-data/raw/hat-p-8-b3001.FIT'


def generate_flat(pathname):
    """
    Load the images in the given folder, and generate a flat field
    correction from the data
    """
    images = glob.glob(pathname)
    fits_images = [fits.open(f) for f in images]
    flat = np.zeros_like(fits_images[0][0].data)

    for im in fits_images:
        flat += im[0].data

    flat = flat / len(fits_images)
    flat = flat / flat.mean()

    return flat


def correct_images(pathname, dark_frame=None, flat_frame=None):
    """
    Load a set of images and correct them using optional dark and flat frames
    """
    images = glob.glob(pathname)
    fits_images = [(path.basename(f), fits.open(f)) for f in images]

    for f, hdulist in fits_images:
        print('Correcting {}'.format(f))
        if flat_frame is not None:
            hdulist[0].data = hdulist[0].data / flat_frame
        fn = CORRECTED_DEST + f
        hdulist.writeto(fn, clobber=True)


if __name__ == '__main__':
    finder.test(DEBUG_IMAGE, snr=3)
    #flat = generate_flat(FLATS)
    #correct_images(IMAGES)
    plt.show()
