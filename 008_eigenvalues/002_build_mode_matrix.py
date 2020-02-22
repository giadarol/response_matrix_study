import numpy as np
from scipy.constants import c as clight

import PyECLOUD.myfilemanager as mfm

from mode_coupling_matrix import CouplingMatrix

# Remember to rescale the beta!!!!

l_min = -5
l_max = 5
m_max = 10
n_phi = 360
n_r = 200
N_max = 50 #199

omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
omega_s = 4.9e-3*omega0
Q_full = 62.27
sigma_b = 0.097057
r_b = 4*sigma_b

a_param = 8./r_b**2

ob = mfm.myloadmat_to_obj('../001_sin_response_scan/response_data.mat')
HH = ob.x_mat
KK = ob.dpx_mat
z_slices = ob.z_slices


MM_obj = CouplingMatrix(z_slices, HH, KK, l_min,
        l_max, m_max, n_phi, n_r, N_max, Q_full, sigma_b, r_b,
        a_param)

# Mode coupling test
strength_scan = np.arange(0, 1.4, 0.02)
Omega_mat = MM_obj.compute_mode_complex_freq(omega_s, rescale_by=strength_scan)

import matplotlib.pyplot as plt
plt.close('all')

mask_unstable = np.imag(Omega_mat) > 1e-1
Omega_mat_unstable = Omega_mat.copy()
Omega_mat_unstable[~mask_unstable] = np.nan

i_mode = -1
mask_mode = np.abs(np.real(Omega_mat)/omega_s - i_mode)<0.5
plt.plot(strength_scan, np.imag(Omega_mat), '.b')
plt.plot(strength_scan, np.imag(Omega_mat), '.b')
Omega_mat_mode = Omega_mat.copy()
Omega_mat_mode[~mask_mode] = np.nan

plt.figure(200)
plt.plot(strength_scan, np.real(Omega_mat)/omega_s, '.b')
plt.plot(strength_scan, np.real(Omega_mat_unstable)/omega_s, '.r')


plt.figure(201)
plt.plot(strength_scan, np.imag(Omega_mat), '.b')
plt.plot(strength_scan, np.imag(Omega_mat_mode), '.g')

plt.figure(300)
plt.plot(np.imag(Omega_mat).flatten(),
        np.real(Omega_mat).flatten()/omega_s, '.b')

plt.show()


