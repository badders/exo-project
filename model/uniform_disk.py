import numpy as np
from astropy import constants as c
from matplotlib import pyplot as plt


# Simulation of jupiter orbiting sun at 0.5 AU
rp = c.R_jup
mp = c.M_jup
sr = c.R_sun
sm = c.M_sun
sl = c.L_sun

d = c.pc
r = 0.2 * c.au

position = np.linspace(-r, r)
flux = np.ones_like(position) * sl / (4 * np.pi * d ** 2)

for i in range(len(position)):
    p = position[i]
    if p > - sr and p < sr:
        flux[i] = flux[i] * (1 - (rp ** 2 / sr ** 2))

plt.plot(position, luminosity)
plt.show()
