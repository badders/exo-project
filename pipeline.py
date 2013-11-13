from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from photometry import finder
import glob
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from os import path
from common.display import show_fits, show_header
from common.dependency import update_required


FLATS = '/Users/tombadran/fits/test-data/flats/*.FIT'
BIAS = '/Users/tombadran/fits/test-data/bias/*.FIT'
IMAGES = '/Users/tombadran/fits/test-data/raw/*.FIT'
CORRECTED_DEST = '/Users/tombadran/fits/test-data/corrected/'


def generate_bias(pathname, force=False):
    """
    Load the images from pathname and generate a bias correction
    """
    BIAS_IMAGE = CORRECTED_DEST + 'bias.fits'

    if force or update_required(BIAS_IMAGE, pathname):
        print('Building new bias image')
        images = glob.glob(pathname)
        fits_images = [fits.open(f) for f in images]
        bias = np.zeros(fits_images[0][0].shape, dtype=np.float64)

        for im in fits_images:
            bias += im[0].data

        bias = bias / len(fits_images)
        fits_images[0].data = bias
        fits_images[0].writeto(BIAS_IMAGE, clobber=True)
    else:
        print('Bias already up to date')

    return BIAS_IMAGE


def generate_flat(pathname, force=False):
    """
    Load the images in the given folder, and generate a flat field
    correction from the data
    """
    FLAT_IMAGE = CORRECTED_DEST + 'flat.fits'

    if force or update_required(FLAT_IMAGE, pathname):
        print('Building new flat image')
        images = glob.glob(pathname)
        fits_images = [fits.open(f) for f in images]
        flat = np.zeros(fits_images[0][0].shape, dtype=np.float64)

        for im in fits_images:
            flat += im[0].data

        flat = flat / flat.mean()
        fits_images[0].data = flat
        fits_images[0].writeto(FLAT_IMAGE, clobber=True)
    else:
        print('Flat already up to date')

    return FLAT_IMAGE


def correct_images(pathname, dark_frame=None, flat_frame=None):
    """
    Load a set of images and correct them using optional dark and flat frames
    """
    images = glob.glob(pathname)
    fits_images = [(path.basename(f), fits.open(f)) for f in images]

    dark = fits.open(dark_frame)[0].data
    flat = fits.open(flat_frame)[0].data

    corrected_images = []
    for f, hdulist in fits_images:
        fn = CORRECTED_DEST + f
        if update_required(fn, dark_frame) and update_required(fn, flat_frame):
            print('Correcting {}'.format(f))

            if dark_frame is not None:
                hdulist[0].data = hdulist[0].data - dark

            if flat_frame is not None:
                hdulist[0].data = hdulist[0].data / flat
            hdulist.writeto(fn, clobber=True)
        else:
            print('{} already corrected'.format(f))
        corrected_images.append(fn)

    return corrected_images


if __name__ == '__main__':
    #finder.test(DEBUG_IMAGE, snr=3)
    bias = generate_bias(BIAS)
    flat = generate_flat(FLATS)
    #show_fits(DEBUG_IMAGE)
    #show_fits(flat)
    im = correct_images(IMAGES, dark_frame=bias, flat_frame=flat)
    show_fits(im[0])
    #show_header(im[0])
    #finder.test(im[0], snr=5)
    plt.show()
