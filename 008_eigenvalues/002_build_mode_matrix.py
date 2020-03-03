import pickle

import numpy as np
from scipy.constants import c as clight

import PyECLOUD.myfilemanager as mfm

from mode_coupling_matrix import CouplingMatrix

# Remember to rescale the beta!!!!

# Reference
l_min = -7
l_max = 7
m_max = 60
n_phi = 3*360
n_r = 3*200
N_max = 199
save_pkl_fname = 'mode_coupling_matrix.pkl'

# # Test
# l_min = -3
# l_max = 3
# m_max = 3
# n_phi = 3*360
# n_r = 3*200
# N_max = 12
# save_pkl_fname = None

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

if save_pkl_fname is not None:
    with open(save_pkl_fname, 'wb') as fid:
        pickle.dump(MM_obj, fid)


# Mode coupling test
strength_scan = np.arange(0, 3.4, 0.02)
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

plt.figure(400)
for ii in range(len(strength_scan)):
    plt.scatter(x=strength_scan[ii]+0*np.imag(Omega_mat[ii, :]),
            y=np.imag(Omega_mat[ii, :]),
            c = np.real(Omega_mat[ii, :])/omega_s,
            vmin=-2, vmax=2, cmap=plt.cm.seismic)
plt.colorbar()


plt.show()


