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
    t_factor = 0.008
    t, uniform = uniform_disk_mandel(0.2 * c.au.value, 2 * c.R_jup.value,
                                     c.M_jup.value, c.R_sun.value, c.M_sun.value,
                                     1000, t_factor)

    _, limb = quadratic_mandel(0.2 * c.au.value, 2 * c.R_jup.value,
                               c.M_jup.value, c.R_sun.value, c.M_sun.value,
                               1000, t_factor)

    _, dumb = uniform_disk(0.2 * c.au.value, 2 * c.R_jup.value,
                           c.M_jup.value, c.R_sun.value, c.M_sun.value,
                           1000, t_factor)

    t = t + abs(t.min())
    plt.plot(t, uniform, 'b', linestyle='--', label='Uniform')
    plt.plot(t, limb, 'r', label='Limb-Dark')
    plt.plot(t, dumb, 'g', linestyle='-.', label='My Model')

    plt.legend(loc='best')
    plt.ylim(0.94, 1.005)
    plt.xlim(t.min(), t.max())
    plt.title('Transit Modelling')
    plt.xlabel('Time / minutes')
    plt.ylabel('Relative Observed Flux')



    plt.show()
