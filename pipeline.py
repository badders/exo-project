# coding : utf-8
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import glob
from os import path

from astropy.io import fits
from astropy.constants import R_sun, R_jup
from common.dependency import update_required
from common.display import show_fits, show_header
from common.gaussian import gaussian2D,  fitgaussian2D
from matplotlib import pyplot as plt
import numpy as np
from photometry import finder
from photometry.aperture import generate_apertures
from photometry.photometry import do_photometry
from model.fitting import fit_quadlimb
from scipy import ndimage, optimize, interpolate
import logging
logging.basicConfig(level=logging.CRITICAL)

FLATS = '/Users/tombadran/fits/chris-data/flats/*.FIT'
BIAS = '/Users/tombadran/fits/chris-data/bias/*.FIT'
IMAGES = '/Users/tombadran/fits/chris-data/raw/*.FIT'
IMAGES2 = '/Users/tombadran/fits/XO2b-2012-01-18/fits/*.fits'

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
    from astropy.time import Time
    times = []

    for im in ims:
        header = fits.open(im)[0].header
        t = Time(header['DATE-OBS'], scale='utc').datetime
        times.append(t)

    for i in range(1, len(times)):
        times[i] = (times[i] - times[0]).total_seconds()
    times[0] = 0

    return times

if __name__ == '__main__':
    bias = generate_bias(BIAS)
    flat = generate_flat(FLATS)
    im = correct_images(IMAGES, flat_frame=flat)
    im2 = correct_images(IMAGES2)
    times = get_times(im)
    times2x = get_times(im2)
    sources = finder.run_sextractor(im[0], DATA_DEST=DATA_DEST)
    sources2 = finder.run_sextractor(im2[0], DATA_DEST=DATA_DEST)

    apertures = generate_apertures(im[0], sources)
    apertures2 = generate_apertures(im2[1], sources2, max_radius=8)

    fig = show_fits(im2[0])
    for i in range(len(apertures2)):
        ap = apertures2[i]
        fig.show_circles(ap.x, ap.y, ap.r)
        plt.annotate(i, xy=(ap.x, ap.y), xytext=(-10, 10),
                     textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='y', alpha=0.2),
                     arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0'))

    # Only use target apeture
    # nb 75 is hat p 20
    apertures = [apertures[75], apertures[70], apertures[65], apertures[49], apertures[105]]
    apertures2 = [apertures2[20], apertures2[15], apertures2[13]]

    phot_data, phot_err = do_photometry(im, apertures, data_store=DATA_DEST+'phot_data.txt', err_store=DATA_DEST+'phot_err.txt', max_radius=20)
    phot_data2, phot_err2 = do_photometry(im2, apertures2, data_store=DATA_DEST+'phot_data2.txt', err_store=DATA_DEST+'phot_err2.txt', max_radius=8, force=False)

    plt.figure()
    plt.plot(phot_data2[0])
    plt.plot(phot_data2[1])
    plt.plot(phot_data2[2])

    fig = show_fits(im2[-1])
    for i in range(len(apertures2)):
        ap = apertures2[i]
        fig.show_circles(ap.x, ap.y, ap.r)
        plt.annotate(i, xy=(ap.x, ap.y), xytext=(-10, 10),
                     textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='y', alpha=0.2),
                     arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0'))
    star = phot_data[0]
    err = phot_err[0]

    ls = []
    errs = []
    for i in range(1, len(phot_data)):
        cal = phot_data[i]
        cal_err = phot_err[i]
        l = star / cal
        err = cal_err / cal
        flux_norm = np.percentile(l, 65)
        l = l / flux_norm
        ls.append(l)
        err = err / flux_norm
        errs.append(err)

    star = np.zeros_like(star)
    err = np.zeros_like(star)

    for l in ls:
        star += l

    for e in errs:
        err += e

    err = err / len(errs)
    star = star / len(ls)

    times, star, err = bin_data(np.array(times), star, err, span=5)
    model_flux, r_p, r_p_err = fit_quadlimb(times, star, err)

    # Convert r_p to Jovian radii
    r_hat = 0.694
    r = (r_p * R_sun * r_hat) / R_jup

    normalise_fac = model_flux[0]
    star = star / normalise_fac
    err = err / normalise_fac
    model_flux = model_flux / normalise_fac

    plt.figure(figsize=(8,6))
    plt.ylabel('Relative Flux')
    plt.xlabel('Time / minutes')
    plt.errorbar(times / 60, star, capsize=0, yerr=err, fmt='ko', label='Data')

    times2 = np.zeros(len(times) + 2)
    times2[1:-1] = times / 60
    times2[0] = 0
    times2[-1] = plt.xlim()[1]

    mflux = np.zeros(len(model_flux) + 2)
    mflux[0] = model_flux[0]
    mflux[1:-1] = model_flux
    mflux[-1] = model_flux[-1]

    plt.plot(times2, mflux, 'b-', label='R={:.2f} RJ'.format(r))

    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig('report/images/chris_curve.pdf')

    star = phot_data2[0]
    err = phot_err2[0]

    ls = []
    errs = []
    for i in range(1, len(phot_data2)):
        cal = phot_data2[i]
        cal_err = phot_err2[i]
        l = star / cal
        err = cal_err / cal
        flux_norm = np.percentile(l, 65)
        l = l / flux_norm
        ls.append(l)
        err = err / flux_norm
        errs.append(err)

    star = np.zeros_like(star)
    err = np.zeros_like(star)

    for l in ls:
        star += l

    for e in errs:
        err += e

    err = err / len(errs)
    star = star / len(ls)

    star[star > 1.1] = np.NaN
    star[star < 0.9] = np.NaN
    times2x, star, err = bin_data(np.array(times2x), star, err, span=3)

    plt.figure()
    plt.plot(np.array(times2x) / 60, star, 'kx')

    #model_flux, r_p, r_p_err = fit_quadlimb(times2x, star, err)

    normalise_fac = model_flux[0]
    star = star / normalise_fac
    err = err / normalise_fac
    model_flux = model_flux / normalise_fac

    plt.ylabel('Relative Flux')
    plt.xlabel('Time / minutes')

    # times2 = np.zeros(len(times2x) + 2)
    # times2[1:-1] = np.array(times2x) / 60
    # times2[0] = 0
    # times2[-1] = plt.xlim()[1]

    # mflux = np.zeros(len(model_flux) + 2)
    # mflux[0] = model_flux[0]
    # mflux[1:-1] = model_flux
    # mflux[-1] = model_flux[-1]

    # plt.plot(times2, mflux, 'b-')

    plt.tight_layout()

    plt.show()
