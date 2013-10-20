# coding : utf-8
"""
Models planetary transits as a uniform disk (both planet and star)
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from astropy import constants as c
from astropy import units as u
from matplotlib import pyplot as plt
from math import *

from quad_limb import quadratic_mandel
from uniform_disk import uniform_disk_mandel, uniform_disk

if __name__ == '__main__':
    # Simulation of large hot jupiter orbiting a star
    d = 0.2 * c.au.value
    rp = 2 * c.R_jup.value
    mp = c.M_jup.value
    rs = c.R_sun.value
    ms = c.M_sun.value
    n = 10000
    t_factor = 0.008

    t, uniform = uniform_disk_mandel(d, rp, mp, rs, ms, n, t_factor)
    _, limb = quadratic_mandel(d, rp, mp, rs, ms, n, t_factor)
    _, dumb = uniform_disk(d, rp, mp, rs, ms, n, t_factor)

    t = (t + abs(t.min())) / 60
    plt.plot(t, dumb, 'k')
    plt.ylim(dumb.min() * 0.99, dumb.max() * 1.01)
    plt.xlim(t.min(), t.max())
    plt.xlabel('Time / minutes')
    plt.ylabel('Relative Observed Flux')
    plt.savefig('../report/images/uniform_disk_model.pdf')

    plt.figure()
    #plt.plot(t, uniform, 'b', linestyle='--', label='Uniform')
    plt.plot(t, limb, 'k', linestyle='-.', label='Quadratic Limb Darkening')
    plt.plot(t, dumb, 'k', linestyle='-', label='Uniform Disk')

    plt.legend(loc='best')
    plt.ylim(0.93, 1.005)
    plt.xlim(t.min(), t.max())
    plt.xlabel('Time / minutes')
    plt.ylabel('Relative Observed Flux')
    plt.savefig('../report/images/model_comparison.pdf')

    plt.show()
