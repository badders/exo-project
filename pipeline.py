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
from photometry.aperture import generate_apertures
from photometry.photometry import do_photometry

from scipy import ndimage
from scipy import optimize

FLATS = '/Users/tombadran/fits/chris-data/flats/*.FIT'
BIAS = '/Users/tombadran/fits/chris-data/bias/*.FIT'
IMAGES = '/Users/tombadran/fits/chris-data/raw/*.FIT'
CORRECTED_DEST = '/Users/tombadran/fits/chris-data/corrected/'
DATA_DEST = '/Users/tombadran/fits/chris-data/corrected/data/'
ALIGNED_DEST = '/Users/tombadran/fits/chris-data/aligned/'


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

    if dark_frame is not None:
        dark = fits.open(dark_frame)[0].data
    if flat_frame is not None:
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
    Return the absolute pixel difference between two images after
    normalisation.
    """
    i1 = im1 - np.median(im1.flatten())
    i2 = im2 - np.median(im2.flatten())
    return np.sqrt(abs(i1 ** 2 - i2 ** 2)).sum()


def im_shift(im, shiftx, shifty, angle=0):
    bg = np.median(im.flatten())
    rotated = im  # ndimage.rotate(im, angle, cval=bg, reshape=False)
    shifted = ndimage.shift(rotated, [shiftx, shifty], cval=bg)
    return shifted


def track_image_shift(im1, im2, guess=np.array([0., 0.])):
    err_func = lambda p: im_diff(im1, im_shift(im2, *p))
    echo = lambda xk: print('Parameters: {}'.format(xk))
    out = optimize.fmin_powell(
        err_func, guess, xtol=0.01, callback=echo, full_output=1)
    params = out[0]
    return params


def align_images(pathname, force=False):
    """
    Load a set of images and align them
    """
    images = glob.glob(pathname)
    corrected_images = [ALIGNED_DEST + path.basename(images[0])]

    fn = ALIGNED_DEST + path.basename(images[0])
    if force or update_required(fn, images[0]):
        hdulist = fits.open(images[0])
        hdulist.writeto(fn, clobber=True)

    guess = np.array([3., -3.])
    for i in range(len(images) - 1):
        f1 = ALIGNED_DEST + path.basename(images[i])
        f2 = images[i + 1]
        f = path.basename(f2)
        fn = ALIGNED_DEST + f
        if force or update_required(fn, f1) or update_required(fn, f2):
            print('Aligning {}'.format(f))

            hdulist1 = fits.open(f1)
            hdulist2 = fits.open(f2)
            params = track_image_shift(
                hdulist1[0].data, hdulist2[0].data, guess)

            hdulist2[0].data = im_shift(hdulist2[0].data, *params)
            hdulist2.writeto(fn, clobber=True)
            guess = params + np.array([3, -3])
        else:
            print('{} already aligned'.format(f))
        corrected_images.append(fn)

    return corrected_images


def bin_data(times, data, error, span=3):
    new_times = []
    new_vals = []
    new_errs = []
    for i in range(0, len(data) // span):
        new_times.append(times[span * i:span * i + span].mean())
        new_vals.append(data[span * i:span * i + span].mean())
        new_errs.append(np.sqrt((error[span * i:span * i + span]**2).sum()) / span)
    return np.array(new_times), np.array(new_vals), np.array(new_errs)


def get_times(ims):
    import datetime
    times = []

    for im in ims:
        header = fits.open(im)[0].header
        t = header['DATE-OBS']
        times.append(datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.000'))

    for i in range(1, len(times)):
        times[i] = (times[i] - times[0]).total_seconds()
    times[0] = 0

    return times

if __name__ == '__main__':
    #finder.test(DEBUG_IMAGE, snr=3)
    bias = generate_bias(BIAS)
    flat = generate_flat(FLATS)
    # show_fits(DEBUG_IMAGE)
    # show_fits(flat)
    # im = correct_images(IMAGES, dark_frame=bias, flat_frame=flat)
    im = correct_images(IMAGES, flat_frame=flat)
    times = get_times(im)
    #show_header(im[0])
    #finder.test(im[0], snr=5)
    #im = align_images(CORRECTED_DEST + 'hat*.FIT')
    sources = finder.run_sextractor(im[0], DATA_DEST=DATA_DEST)

    apertures = generate_apertures(im[0], sources)
    #apertures = filter_saturated(im[0], apertures)

    fig = show_fits(im[0])
    for i in range(len(apertures)):
        ap = apertures[i]
        fig.show_circles(ap.x, ap.y, ap.r)
        plt.annotate(i, xy=(ap.x, ap.y), xytext=(-10, 10),
                     textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='y', alpha=0.2),
                     arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0'))

    # Hack to only use target apeture
    # nb 75 is hat p 20
    apertures = [apertures[75], apertures[70], apertures[65], apertures[49], apertures[105]]

    phot_data, phot_err = do_photometry(im, apertures)
    plt.figure()

    star = phot_data[0]
    err = phot_err[0]

    ls = []
    errs = []
    for i in range(1, len(phot_data)):
        l = star / phot_data[i]
        err = phot_err[i] / phot_data[i]
        ls.append(l / l.mean())
        errs.append(err / l.mean())

    star = np.zeros_like(star)
    err = np.zeros_like(star)

    for l in ls:
        star += l

    for e in errs:
        err += e

    err = err / len(errs)
    star = star / len(ls)

    times, star, err = bin_data(np.array(times), star, err, span=5)
    #star = -2.5 * np.log10(star)

    #star += 11.3 - star.mean()
    # err = err / c1
    #plt.plot(np.array(times) / 60, star, 'g.')

    plt.ylabel('Flux')
    plt.xlabel('Time / minutes')

    plt.errorbar(times / 60, star, capsize=0, yerr=err, fmt='ko')
    plt.tight_layout()
    # plt.plot(phot_data[0] / phot_data[2])
    # plt.plot(phot_data[0] / phot_data[3])

    # fig = show_fits(im[-1])
    # ap = apertures[0]
    # print(ap)
    # fig.show_circles(ap.x, ap.y, ap.r)

    #plt.plot(sources[0][1]['X_IMAGE'], sources[0][1]['Y_IMAGE'], 'ro', markersize=4)
    #fig = show_fits(im[-1])

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
