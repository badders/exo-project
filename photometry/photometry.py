from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from astropy.io import fits
from common.gaussian import fitgaussian2D
import photutils

def do_photometry(ims, aps, max_radius=60):
    import math
    phot_data = np.zeros((len(aps), len(ims)))
    phot_err = np.zeros_like(phot_data)

    xs = np.zeros((len(aps), len(ims)), dtype=np.float64)
    ys = np.zeros_like(xs)

    for i in range(len(ims)):
        data = fits.open(ims[i])[0].data
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

    return phot_data, phot_err
