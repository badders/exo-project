
# coding : utf-8
"""
Models planetary transits as a uniform disk (both planet and star)
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from astropy import constants as c
from math import *

def overlap_area(r1, r2, d):
    def segment(r1, r2, d):
        return r1**2 * acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    return segment(r1, r2, d) + segment(r2, r1, d) - 0.5 * \
        np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) *
                (d + r1 + r2))

def uniform_disk(r, rp, mp, rs, ms, n=1000, t_factor=0.05):
    """
    Very simple model of planet transit

    Arguments:
    r -- Planet orbital distance (m)
    rp -- Radius of planet (m)
    mp -- Mass of planet (kg)
    rs -- Radius of star (m)
    ms -- Mass of star (kg)
    n -- Number of timesteps
    t_factor -- Amount of period around which to plot
    """
    # Generate Positions for small section of orbit
    T = np.sqrt(4 * pi**2 * r**3 / (c.G * ms))
    w = np.sqrt(c.G * ms / r**3)
    t = np.linspace(-t_factor * T, t_factor * T, num=n)
    xpos = r * np.sin(w * t)
    observed = np.zeros_like(xpos)

    # Loop over positions and check for occlusion
    i = 0
    for x in xpos:
        if x - rp > -rs and x + rp < rs:
            # Full occlusion
            observed[i] = (rs**2 - rp**2) / (rs ** 2)
        elif (x + rp > -rs and x - rp < -rs) or (x + rp > rs and x - rp < rs):
            # Partial occlusion 
            observed[i] = (pi * rs**2 - overlap_area(rp, rs, abs(x))) / (pi * rs**2)
        else:
            observed[i] = 1
        i += 1

    return t, observed

def uniform_disk_mandel(r, rp, mp, rs, ms, n=1000, t_factor=0.05):
    """
    Very simple model of planet transit

    Arguments:
    r -- Planet orbital distance (m)
    rp -- Radius of planet (m)
    mp -- Mass of planet (kg)
    rs -- Radius of star (m)
    ms -- Mass of star (kg)
    n -- Number of timesteps
    t_factor -- Amount of period around which to plot
    """
    # Generate Positions for small section of orbit
    T = np.sqrt(4 * pi**2 * r**3 / (c.G * ms))
    w = np.sqrt(c.G * ms / r**3)
    t = np.linspace(-t_factor * T, t_factor * T, num=n)
    xpos = r * np.sin(w * t)
    # Loop over positions and check for occlusion
    z = xpos / rs
    p = rp / rs

    ratio = occultuniform(z, p)
    return t, ratio

def occultuniform(z, p, complement=False):
    """Uniform-disk transit light curve (i.e., no limb darkening).

    :INPUTS:
       z -- scalar or sequence; positional offset values of planet in
            units of the stellar radius.

       p -- scalar;  planet/star radius ratio.

       complement : bool
         If True, return (1 - occultuniform(z, p))

    :SEE ALSO:  :func:`t2z`, :func:`occultquad`, :func:`occultnonlin_small`
    """
    z = np.abs(np.array(z,copy=True))
    fsecondary = np.zeros(z.shape,float)
    if p < 0:
        pneg = True
        p = np.abs(p)
    else:
        pneg = False

    p2 = p**2

    if len(z.shape)>0: # array entered
        i1 = (1+p)<z
        i2 = (np.abs(1-p) < z) * (z<= (1+p))
        i3 = z<= (1-p)
        i4 = z<=(p-1)
        #print i1.sum(),i2.sum(),i3.sum(),i4.sum()

        z2 = z[i2]**2
        acosarg1 = (p2+z2-1)/(2.*p*z[i2])
        acosarg2 = (1-p2+z2)/(2*z[i2])
        acosarg1[acosarg1 > 1] = 1.  # quick fix for numerical precision errors
        acosarg2[acosarg2 > 1] = 1.  # quick fix for numerical precision errors
        k0 = np.arccos(acosarg1)
        k1 = np.arccos(acosarg2)
        k2 = 0.5*np.sqrt(4*z2-(1+z2-p2)**2)

        fsecondary[i1] = 0.
        fsecondary[i2] = (1./np.pi)*(p2*k0 + k1 - k2)
        fsecondary[i3] = p2
        fsecondary[i4] = 1.

        if not (i1+i2+i3+i4).all():
            print("warning -- some input values not indexed!")
        if (i1.sum()+i2.sum()+i3.sum()+i4.sum() != z.size):
            print("warning -- indexing didn't get the right number of values")
            #pdb.set_trace()
        

    else:  # scalar entered
        if (1+p)<=z:
            fsecondary = 0.
        elif (np.abs(1-p) < z) * (z<= (1+p)):
            z2 = z**2
            k0 = np.arccos((p2+z2-1)/(2.*p*z))
            k1 = np.arccos((1-p2+z2)/(2*z))
            k2 = 0.5*np.sqrt(4*z2-(1+z2-p2)**2)
            fsecondary = (1./np.pi)*(p2*k0 + k1 - k2)
        elif z<= (1-p):
            fsecondary = p2
        elif z<=(p-1):
            fsecondary = 1.
        
    if pneg:
        fsecondary *= -1

    if complement:
        return fsecondary
    else:
        return 1. - fsecondary
