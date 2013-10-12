
# coding : utf-8
"""
Models planetary transits as a uniform disk (both planet and star)
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from astropy import constants as c
from astropy import units as u
from matplotlib import pyplot as plt
from math import *

def overlap_area(r1, r2, d):
    def segment(r1, r2, d):
        return r1**2 * acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    return segment(r1, r2, d) + segment(r2, r1, d) - 0.5 * \
        np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) *
                (d + r1 + r2))


def uniform_disk(r, rp, mp, rs, ms, ls, d, n=1000, t_factor=0.05):
    """
    Very simple model of planet transit

    Arguments:
    r -- Planet orbital distance (m)
    rp -- Radius of planet (m)
    mp -- Mass of planet (kg)
    rs -- Radius of star (m)
    ms -- Mass of star (kg)
    ls -- Luminosity of star (W)
    d -- Observer distance to star (parsecs)
    n -- Number of timesteps
    t_factor -- Amount of period around which to plot
    """
    plt.figure()
    # Generate Positions for small section of orbit
    T = np.sqrt(4 * pi**2 * r**3 / (c.G * ms))
    w = np.sqrt(c.G * ms / r**3)
    t = np.linspace(-t_factor * T, t_factor * T, num=n)
    xpos = r * np.sin(w * t)
    flux = np.ones_like(xpos) * ls / (4 * np.pi * d ** 2)
    observed = np.zeros_like(flux)

    # Loop over positions and check for occlusion
    i = 0
    for x, f in zip(xpos, flux):
        if x - rp > -rs and x + rp < rs:
            # Full occlusion
            observed[i] = f * (rs**2 - rp**2) / (rs ** 2)
        elif (x + rp > -rs and x - rp < -rs) or (x + rp > rs and x - rp < rs):
            # Partial occlusion 
            observed[i] = f * (pi * rs**2 - overlap_area(rp, rs, abs(x))) / (pi * rs**2)
        else:
            observed[i] = f
        i += 1

    return t, observed

if __name__ == '__main__':
    # Simulation of large hot jupiter orbiting a star
    t_factor = 0.0065
    t, observed = uniform_disk(0.2 * c.au.value, 2 * c.R_jup.value,
                               c.M_jup.value, c.R_sun.value, c.M_sun.value,
                               c.L_sun.value, c.pc.value, 100, t_factor)

    t = t * u.s.to(u.min)
    obs_pc = observed / observed.max() * 100
    plt.plot(t - t.min(), obs_pc, 'bx')
    pad = (obs_pc.max() - obs_pc.min()) / 20
    plt.ylim(obs_pc.min() - pad, obs_pc.max() + pad)
    plt.xlim(t.min() - t.min(), t.max() - t.min() )
    plt.title('Very Simple Model')
    plt.xlabel('Time / minutes')
    plt.ylabel('Relative Measured Flux Percentage')

    plt.show()
