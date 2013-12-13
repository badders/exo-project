from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import glob

import numpy as np
from matplotlib import pyplot as plt
from os import path
from astropy.io import fits
from common.display import show_fits, show_header
from common.dependency import update_required
from common.gaussian import gaussian2D,  fitgaussian2D
from photometry import finder

from scipy import ndimage

FLATS = '/Users/tombadran/fits/test-data/flats/*.FIT'
BIAS = '/Users/tombadran/fits/test-data/bias/*.FIT'
IMAGES = '/Users/tombadran/fits/test-data/raw/*.FIT'
CORRECTED_DEST = '/Users/tombadran/fits/test-data/corrected/'
DATA_DEST = '/Users/tombadran/fits/test-data/corrected/data/'


def generate_bias(pathname, force=False):
    """
    Load the images from pathname and generate a bias correction
    """
    BIAS_IMAGE = CORRECTED_DEST + 'bias.fits'

    if force or update_required(BIAS_IMAGE, pathname):
        images = glob.glob(pathname)
        fits_images = [fits.open(f) for f in images]
        print('Building new bias image')
        bias = np.zeros(fits_images[0][0].shape, dtype=np.float64)

        for im in fits_images:
            bias += im[0].data

        bias = bias / len(fits_images)
        fits_images[0][0].data = bias
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
        images = glob.glob(pathname)
        fits_images = [fits.open(f) for f in images]
        print('Building new flat image')
        flat = np.zeros(fits_images[0][0].shape, dtype=np.float64)

        for im in fits_images:
            flat += im[0].data

        flat = flat / flat.mean()

        print('Gaussian fitting the flat image')
        params = fitgaussian2D(flat)
        flat = gaussian2D(*params)(*np.indices(flat.shape))

        fits_images[0][0].data = flat
        fits_images[0].writeto(FLAT_IMAGE, clobber=True)
    else:
        print('Flat already up to date')

    return FLAT_IMAGE


def correct_images(pathname, dark_frame=None, flat_frame=None, force=False):
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
        if force or update_required(fn, dark_frame) or update_required(fn, flat_frame):
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


def im_diff(im1, im2):
    """
    Return the absolute pixel difference between two images after normalisation.
    """
    return abs((im1 / im1.mean()) - (im2 / im2.mean())).sum()

def im_shift(im, shift, angle):
    bg = np.median(im.flatten())
    rotated = ndimage.rotate(im, angle, cval=bg, reshape=False)
    shifted = ndimage.shift(rotated, shift, cval=bg)
    return shifted

def do_photometry(data, star, cal_stars):
    fluxes = []
    errs = []
    times = []
    x0 = data[0][1]['X_IMAGE'][star]
    y0 = data[0][1]['Y_IMAGE'][star]
    aperture_r = sources[0][1]['A_IMAGE'] * sources[0][1]['KRON_RADIUS'][star]

    print(x0, y0)
    for time, image in data:
        flux = image['FLUX_BEST'][star]
        err = image['FLUXERR_BEST'][star]
        x1 = image['X_IMAGE'][star]
        y1 = image['Y_IMAGE'][star]
        print(x1, y1, np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

        fluxes.append(flux)
        errs.append(err)
        times.append(time)

    times = np.arange(len(fluxes))
    return times, fluxes, errs


if __name__ == '__main__':
    #finder.test(DEBUG_IMAGE, snr=3)
    bias = generate_bias(BIAS)
    flat = generate_flat(FLATS)
    # show_fits(DEBUG_IMAGE)
    # show_fits(flat)
    im = correct_images(IMAGES, dark_frame=bias, flat_frame=flat)
    # show_header(im[0])
    #finder.test(im[0], snr=5)

    sources = finder.run_sextractor(im, DATA_DEST=DATA_DEST)

    fig = show_fits(im[0])
    plt.plot(sources[0][1]['X_IMAGE'] + 1, sources[0][1]['Y_IMAGE'] + 1, 'ro', markersize=4)

    print(im[0])

    # fig = show_fits(im_shift(fits.open(im[0])[0].data, [100, 200], 15))
    # counter = 0
    # for x, y in zip(sources[0][1]['X_IMAGE'], sources[0][1]['Y_IMAGE']):
    #     plt.annotate(counter, xy=(x, y), xytext=(-10, 10),
    #                  textcoords='offset points', ha='right', va='bottom',
    #                  bbox=dict(boxstyle='round,pad=0.5', fc='y', alpha=0.2),
    #                  arrowprops=dict(arrowstyle='->',
    #                                  connectionstyle='arc3,rad=0'))
    #     counter += 1)
    # fig.show_circles(sources[0][1]['X_IMAGE'] + 1,
    #                  sources[0][1]['Y_IMAGE'] + 1,
    #                  sources[0][1]['A_IMAGE'] * sources[0][1]['KRON_RADIUS'],
    #                  edgecolor='y', linewidth=1)

    # times, fluxes, errs = do_photometry(sources, 91, [])
    # plt.figure()
    #plt.errorbar(times, fluxes, yerr=errs)
    plt.show()
