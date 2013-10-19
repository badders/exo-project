"""
Models planetary transits using a quardratic approximation of limb darkening
From Mandel & Agol 2002
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from uniform_disk import occultuniform
from astropy import constants as c
from math import *

eps = np.finfo(float).eps
zeroval = eps*1e6

def quadratic_mandel(r, rp, mp, rs, ms, n=1000, t_factor=0.05, gamma=[0.3, 0.7]):
    """
    Quadratic limb darkening model from Mandel & Agol 2002

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
    z = abs(xpos / rs)
    p = rp / rs

    ratio = occultquad(z, p, gamma)
    return t, ratio

def ellke(k):
    """Compute Hasting's polynomial approximation for the complete
    elliptic integral of the first (ek) and second (kk) kind.

    :INPUTS:
       k -- scalar or Numpy array
      
    :OUTPUTS:
       ek, kk

    :NOTES:
       Adapted from the IDL function of the same name by J. Eastman (OSU).
       """
    # 2011-04-19 09:15 IJC: Adapted from J. Eastman's IDL code.
    
    m1 = 1. - k**2
    logm1 = np.log(m1)

    # First kind:
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639

    ee1 = 1. + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ee2 = m1 * (b1 + m1*(b2 + m1*(b3 + m1*b4))) * (-logm1)
    
    # Second kind:     
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    ek1 = a0 + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ek2 = (b0 + m1*(b1 + m1*(b2 + m1*(b3 + m1*b4)))) * logm1

    return ee1 + ee2, ek1 - ek2

def ellpic_bulirsch(n, k, tol=1000*eps, maxiter=1e4):
    """Compute the complete elliptical integral of the third kind
    using the algorithm of Bulirsch (1965).

    :INPUTS:
       n -- scalar or Numpy array

       k-- scalar or Numpy array

    :NOTES:
       Adapted from the IDL function of the same name by J. Eastman (OSU).
       """
    if not hasattr(n,'__iter__'):
        n = np.array([n])
    if not hasattr(k,'__iter__'):
        k = np.array([k])

    if len(n)==0 or len(k)==0:
        return np.array([])

    kc = np.sqrt(1. - k**2)
    p = n + 1.
        
    # Initialize:
    m0 = np.array(1.)
    c = np.array(1.)
    p = np.sqrt(p)
    d = 1./p
    e = kc.copy()

    outsideTolerance = True
    iter = 0
    while outsideTolerance and iter<maxiter:
        f = c.copy()
        c = d/p + c
        g = e/p
        d = 2. * (f*g + d)
        p = g + p; 
        g = m0.copy()
        m0 = kc + m0
        if ((np.abs(1. - kc/g)) > tol).any():
            kc = 2. * np.sqrt(e)
            e = kc * m0
            iter += 1
        else:
            outsideTolerance = False

    return .5 * np.pi * (c*m0 + d) / (m0 * (m0 + p))

def occultquad(z,p0, gamma, retall=False):
    """Quadratic limb-darkening light curve; cf. Section 4 of Mandel & Agol (2002).

    :INPUTS:
        z -- sequence of positional offset values

        p0 -- planet/star radius ratio

        gamma -- two-sequence.
           quadratic limb darkening coefficients.  (c1=c3=0; c2 =
           gamma[0] + 2*gamma[1], c4 = -gamma[1]).  If only a single
           gamma is used, then you're assuming linear limb-darkening.

    :OPTIONS:
        retall -- bool.  
           If True, in addition to the light curve return the
           uniform-disk light curve, lambda^d, and eta^d parameters.
           Using these quantities allows for quicker model generation
           with new limb-darkening coefficients -- the speed boost is
           roughly a factor of 50.  See the second example below.

    :EXAMPLE:
       ::

         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 100)
         gammavals = [[0., 0.], [1., 0.], [2., -1.]]
         figure()
         for gammas in gammavals:
             f = transit.occultquad(z, 0.1, gammas)
             plot(z, f)

       ::

         # Calculate the same geometric transit with two different
         #    sets of limb darkening coefficients:
         from pylab import *
         import transit
         p, b = 0.1, 0.5
         x = (arange(300.)/299. - 0.5)*2.
         z = sqrt(x**2 + b**2)
         gammas = [.25, .75]
         F1, Funi, lambdad, etad = transit.occultquad(z, p, gammas, retall=True)

         gammas = [.35, .55]
         F2 = 1. - ((1. - gammas[0] - 2.*gammas[1])*(1. - F1) + 
            (gammas[0] + 2.*gammas[1])*(lambdad + 2./3.*(p > z)) + gammas[1]*etad) / 
            (1. - gammas[0]/3. - gammas[1]/6.)
         figure()
         plot(x, F1, x, F2)
         legend(['F1', 'F2'])
         

    :SEE ALSO:
       :func:`t2z`, :func:`occultnonlin_small`, :func:`occultuniform`

    :NOTES:
       In writing this I relied heavily on the occultquad IDL routine
       by E. Agol and J. Eastman, especially for efficient computation
       of elliptical integrals and for identification of several
       apparent typographic errors in the 2002 paper (see comments in
       the source code).

       From some cursory testing, this routine appears about 9 times
       slower than the IDL version.  The difference drops only
       slightly when using precomputed quantities (i.e., retall=True).
       A large portion of time is taken up in :func:`ellpic_bulirsch`
       and :func:`ellke`, but at least as much is taken up by this
       function itself.  More optimization (or a C wrapper) is desired!
    """
    # Initialize:
    gamma = np.array(gamma, copy=True)
    if gamma.size < 2:  # Linear limb-darkening
        gamma = np.array([gamma.ravel(), [0.]])
    z = np.array(z, copy=True)
    lambdad = np.zeros(z.shape, float)
    etad = np.zeros(z.shape, float)
    F = np.ones(z.shape, float)

    p = np.abs(p0) # Save the original input


    # Define limb-darkening coefficients:
    c2 = gamma[0] + 2 * gamma[1]
    c4 = -gamma[1]

    # Test the simplest case (a zero-sized planet):
    if p==0:
        if retall:
            ret = np.ones(z.shape, float), np.ones(z.shape, float), \
                  np.zeros(z.shape, float), np.zeros(z.shape, float)
        else:
            ret = np.ones(z.shape, float)
        return ret

    # Define useful constants:
    fourOmega = 1. - gamma[0]/3. - gamma[1]/6. # Actually 4*Omega
    a = (z - p)**2
    b = (z + p)**2
    k = 0.5 * np.sqrt((1. - a) / (z * p))
    p2 = p**2
    z2 = z**2

    # Define the many necessary indices for the different cases:
    i01 = (p > 0) * (z >= (1. + p))
    i02 = (p > 0) * (z > (.5 + np.abs(p - 0.5))) * (z < (1. + p))
    i03 = (p > 0) * (p < 0.5) * (z > p) * (z < (1. - p))
    i04 = (p > 0) * (p < 0.5) * (z == (1. - p))
    i05 = (p > 0) * (p < 0.5) * (z == p)
    i06 = (p == 0.5) * (z == 0.5)
    i07 = (p > 0.5) * (z == p)
    i08 = (p > 0.5) * (z >= np.abs(1. - p)) * (z < p)
    i09 = (p > 0) * (p < 1) * (z > 0) * (z < (0.5 - np.abs(p - 0.5)))
    i10 = (p > 0) * (p < 1) * (z == 0)
    i11 = (p > 1) * (z >= 0.) * (z < (p - 1.))
    
    # Lambda^e is easy:
    lambdae = 1. - occultuniform(z, p)  

    # Lambda^e and eta^d are more tricky:
    # Simple cases:
    lambdad[i01] = 0.
    etad[i01] = 0.

    lambdad[i06] = 1./3. - 4./9./np.pi
    etad[i06] = 3./32.

    lambdad[i11] = 1.
    # etad[i11] = 1.  # This is what the paper says
    etad[i11] = 0.5 # Typo in paper (according to J. Eastman)


    # Lambda_1:
    ilam1 = i02 + i08
    q1 = p2 - z2[ilam1]
    ## This is what the paper says:
    #ellippi = ellpic_bulirsch(1. - 1./a[ilam1], k[ilam1])
    # ellipe, ellipk = ellke(k[ilam1])

    # This is what J. Eastman's code has:

    # 2011-04-24 20:32 IJMC: The following codes act funny when
    #                        sqrt((1-a)/(b-a)) approaches unity.
    qq = np.sqrt((1. - a[ilam1]) / (b[ilam1] - a[ilam1]))
    ellippi = ellpic_bulirsch(1./a[ilam1] - 1., qq)
    ellipe, ellipk = ellke(qq)
    lambdad[i02 + i08] = (1./ (9.*np.pi*np.sqrt(p*z[ilam1]))) * \
        ( ((1. - b[ilam1])*(2*b[ilam1] + a[ilam1] - 3) - \
               3*q1*(b[ilam1] - 2.)) * ellipk + \
              4*p*z[ilam1]*(z2[ilam1] + 7*p2 - 4.) * ellipe - \
              3*(q1/a[ilam1])*ellippi)

    # Lambda_2:
    ilam2 = i03 + i09
    q2 = p2 - z2[ilam2]

    ## This is what the paper says:
    #ellippi = ellpic_bulirsch(1. - b[ilam2]/a[ilam2], 1./k[ilam2])
    # ellipe, ellipk = ellke(1./k[ilam2])

    # This is what J. Eastman's code has:
    ellippi = ellpic_bulirsch(b[ilam2]/a[ilam2] - 1, np.sqrt((b[ilam2] - a[ilam2])/(1. - a[ilam2])))
    ellipe, ellipk = ellke(np.sqrt((b[ilam2] - a[ilam2])/(1. - a[ilam2])))

    lambdad[ilam2] = (2. / (9*np.pi*np.sqrt(1.-a[ilam2]))) * \
        ((1. - 5*z2[ilam2] + p2 + q2**2) * ellipk + \
             (1. - a[ilam2])*(z2[ilam2] + 7*p2 - 4.) * ellipe - \
             3*(q2/a[ilam2])*ellippi)


    # Lambda_3:
    #ellipe, ellipk = ellke(0.5/ k)  # This is what the paper says
    ellipe, ellipk = ellke(0.5/ p)  # Corrected typo (1/2k -> 1/2p), according to J. Eastman
    lambdad[i07] = 1./3. + (16.*p*(2*p2 - 1.)*ellipe - 
                                (1. - 4*p2)*(3. - 8*p2)*ellipk / p) / (9*np.pi)


    # Lambda_4
    #ellipe, ellipk = ellke(2. * k)  # This is what the paper says
    ellipe, ellipk = ellke(2. * p)  # Corrected typo (2k -> 2p), according to J. Eastman
    lambdad[i05] = 1./3. + (2./(9*np.pi)) * (4*(2*p2 - 1.)*ellipe + (1. - 4*p2)*ellipk)

    # Lambda_5
    ## The following line is what the 2002 paper says:
    #lambdad[i04] = (2./(3*np.pi)) * (np.arccos(1 - 2*p) - (2./3.) * (3. + 2*p - 8*p2))
    # The following line is what J. Eastman's code says:
    lambdad[i04] = (2./3.) * (np.arccos(1. - 2*p)/np.pi - \
                                  (2./(3*np.pi)) * np.sqrt(p * (1.-p)) * \
                                  (3. + 2*p - 8*p2) - \
                                  float(p > 0.5))

    # Lambda_6
    lambdad[i10] = -(2./3.) * (1. - p2)**1.5

    # Eta_1:
    kappa0 = np.arccos((p2+z2[i02 + i07 + i08]-1)/(2.*p*z[i02 + i07 + i08]))
    kappa1 = np.arccos((1-p2+z2[i02 + i07 + i08])/(2*z[i02 + i07 + i08]))
    etad[i02 + i07 + i08] = \
        (0.5/np.pi) * (kappa1 + kappa0*p2*(p2 + 2*z2[i02 + i07 + i08]) - \
                        0.25*(1. + 5*p2 + z2[i02 + i07 + i08]) * \
                        np.sqrt((1. - a[i02 + i07 + i08]) * (b[i02 + i07 + i08] - 1.))) 


    # Eta_2:
    etad[i03 + i04 + i05 + i09 + i10] = 0.5 * p2 * (p2 + 2. * z2[i03 + i04 + i05 + i09 + i10])
    

    F = 1. - ((1. - c2) * lambdae + \
                  c2 * (lambdad + (2./3.) * (p > z).astype(float)) - \
                  c4 * etad) / fourOmega

    #pdb.set_trace()
    if retall:
        ret = F, lambdae, lambdad, etad
    else:
        ret = F

    #pdb.set_trace()
    return ret
