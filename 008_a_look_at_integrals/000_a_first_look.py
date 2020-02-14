import numpy as np

R_bkt = 0.75
r_b = 9e-2
#a = 8/r_b**2
a = 1/r_b**2
nn = 6

r = np.linspace(0, 3*r_b, 100)
phi = np.linspace(0, 2*np.pi, 3600)

RR, PP = np.meshgrid(r, phi)

exp1 = np.exp(-a*RR*RR*(1. - np.cos(PP)**2/(2*a*r_b**2)))
exp2 = np.exp(1j * 2 * np.pi * nn / R_bkt * np.cos(PP))
prod = exp1*exp2

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
plt.plot(phi, np.real(prod[:, ::10]))

plt.show()
