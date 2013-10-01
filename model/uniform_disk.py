
import numpy as np
from astropy import constants as c
from astropy import units as u
from matplotlib import pyplot as plt
from math import *


def simple_model(r, rp, mp, rs, ms, ls, d, n=1000, t_factor=0.05):
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
        if x > -rs and x < rs:
            observed[i] = f * (rs**2 - rp**2) / (rs ** 2)
        else:
            observed[i] = f
        i += 1

    t = (t + t_factor * T) * u.s.to(u.min)
    obs_pc = observed / observed.max() * 100
    plt.plot(t, obs_pc)
    plt.ylim(obs_pc.min() - 0.05, obs_pc.max() + 0.05)
    plt.xlim(t.min(), t.max())
    plt.title('Very Simple Model')
    plt.xlabel('Time / minutes')
    plt.ylabel('Relative Measured Flux Percentage')


if __name__ == '__main__':
    # Simulation of jupiter orbiting sun at 0.2 AU, viewed from a parsec

    simple_model(0.2 * c.au, c.R_jup, c.M_jup, c.R_sun, c.M_sun, c.L_sun, c.pc)
    plt.show()
