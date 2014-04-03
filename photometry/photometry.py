# coding : utf-8
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from astropy.io import fits
from common.gaussian import fitgaussian2D, gaussian2D
from common.dependency import update_required
import numpy as np
import photutils
import math


def do_photometry(ims, aps, max_radius=60, data_store=None, err_store=None, force=False):
    if data_store is not None and err_store is not None and not force:
        if not update_required(data_store, ims):
            print('Restoring photometry data from files')
            phot_data = np.loadtxt(data_store)
            phot_err = np.loadtxt(err_store)
            return phot_data, phot_err

    phot_data = np.zeros((len(aps), len(ims)))
    phot_err = np.zeros_like(phot_data)

    xs = np.zeros((len(aps), len(ims)), dtype=np.float64)
    ys = np.zeros_like(xs)

    for i in range(len(ims)):
        source = fits.open(ims[i])
        data = source[0].data
        header = source[0].header

        try:
            egain = header['EGAIN']
        except KeyError:
            egain = 1.

        data = data * egain

        print('Performing photometry on {}'.format(ims[i]))

        for j in range(len(aps)):
            ap = aps[j]

            update_fail = False

            # Need to recenter aperture
            y_min = math.floor(max(ap.y - max_radius, 0))
            y_max = math.floor(min(ap.y + max_radius, data.shape[0] - 1))
            x_min = math.ceil(max(ap.x - max_radius, 0))
            x_max = math.ceil(min(ap.x + max_radius, data.shape[1] - 1))

            if y_min < y_max - 1 and x_min < x_max - 1:
                star = data[y_min:y_max, x_min:x_max]

                params = fitgaussian2D(star)
                dx = params[2]
                dy = params[1]

                if j == 2 and i == 1 and False:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.imshow(star)
                    xlim = plt.xlim()
                    xlim = (xlim[0] + 15, xlim[1]-15)
                    ylim = plt.ylim()
                    ylim = (ylim[0] - 15, ylim[1]+15)
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                    plt.tight_layout()
                    plt.figure()
                    plt.imshow(star)
                    plt.contour(gaussian2D(*params)(*np.indices(star.shape)), linewidths=2)
                    plt.hlines(dx-1, *plt.ylim(), linewidths=2)
                    plt.vlines(dy+1, *plt.xlim(), linewidths=2)
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                    plt.tight_layout()

                    return phot_data, phot_err


                nx = dx + x_min
                ny = dy + y_min

                if abs(nx - ap.x) > max_radius:
                    nx = ap.x

                if abs(ny - ap.y) > max_radius:
                    ny = ap.y

                xs[j][i] = nx
                ys[j][i] = ny

                ap = ap._replace(x=nx, y=ny)
                aps[j] = ap

        axs = np.array([ap.x for ap in aps])
        ays = np.array([ap.y for ap in aps])
        rs = np.array([ap.r for ap in aps])
        br1s = np.array([ap.br1 for ap in aps])
        br2s = np.array([ap.br2 for ap in aps])

        rawflux, rawflux_err = photutils.aperture_circular(data, axs, ays, rs, error=np.sqrt(data))

        bkg = photutils.annulus_circular(data, axs, ays, br1s, br2s)
        ap_areas = np.pi * rs ** 2
        bkg_areas = np.pi * (br2s ** 2 - br1s ** 2)
        flux = rawflux - bkg * ap_areas / bkg_areas

        phot_data[:,i] = flux
        phot_err[:,i] = rawflux_err

    if data_store is not None and err_store is not None:
        np.savetxt(data_store, phot_data)
        np.savetxt(err_store, phot_err)

    return phot_data, phot_err
