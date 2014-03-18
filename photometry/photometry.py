from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from astropy.io import fits
from common.gaussian import fitgaussian2D


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
            else:
                update_fail = True

            if update_fail:
                phot_data[j][i] = np.NaN
            else:
                y0 = int(math.floor(max(ap.y - ap.br2, 0)))
                y1 = int(math.floor(min(ap.y + ap.br2, data.shape[0] - 1)))
                x0 = int(math.ceil(max(ap.x - ap.br2, 0)))
                x1 = int(math.ceil(min(ap.x + ap.br2, data.shape[1] - 1)))

                star = data[y0:y1, x0:x1]

                my, mx = np.ogrid[y0:y1, x0:x1]

                ap_mask = (mx-ap.x)**2 + (my-ap.y)**2 <= ap.r**2
                br2_mask = (mx-ap.x)**2 + (my-ap.y)**2 <= ap.br2**2
                br1_mask = (mx-ap.x)**2 + (my-ap.y)**2 <= ap.br1**2

                br2_mask[br1_mask] = 0

                bg = (star * br2_mask).sum() / br2_mask.sum()
                star = star * ap_mask
                flux = star.sum() - ap_mask.sum() * bg
                flux_err = np.sqrt((np.sqrt(star)**2).sum())

                phot_data[j][i] = flux
                phot_err[j][i] = flux_err

    return phot_data, phot_err
