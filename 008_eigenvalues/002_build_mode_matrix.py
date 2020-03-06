import pickle

import numpy as np
from scipy.constants import c as clight

import PyECLOUD.myfilemanager as mfm

from mode_coupling_matrix import CouplingMatrix

# Remember to rescale the beta!!!!

# # Reference
# l_min = -7
# l_max = 7
# m_max = 30
# n_phi = 3*360
# n_r = 3*200
# N_max = 49
# n_tail_cut = 0
# save_pkl_fname = 'mode_coupling_matrix.pkl'
# response_matrix_file = '../001_sin_response_scan/response_data_processed.mat'
# z_strength_file = '../001a_sin_response_scan_unperturbed/linear_strength.mat'
# pool_size = 4

# Test
l_min = -7
l_max = 7
m_max = 3
n_phi = 3*360
n_r = 3*200
N_max = 50
save_pkl_fname = None
n_tail_cut = 0
response_matrix_file = '../001_sin_response_scan/response_data_processed.mat'
z_strength_file = '../001a_sin_response_scan_unperturbed/linear_strength.mat'
detuning_fit_order = 0
pool_size = 4

omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
omega_s = 4.9e-3*omega0
Q_full = 62.27
sigma_b = 0.097057
r_b = 4*sigma_b

a_param = 8./r_b**2


# Prepare response matrix
ob = mfm.myloadmat_to_obj(response_matrix_file)
HH = ob.x_mat
KK = ob.dpx_mat

if n_tail_cut > 0:
    KK[:, :n_tail_cut] = 0.
    KK[:, -n_tail_cut:] = 0.
z_slices = ob.z_slices

# Load detuning with z
if detuning_fit_order > 0:
    obdet = mfm.myloadmat_to_obj(z_strength_file)
    z_slices = obdet.z_slices
    p = np.polyfit(obdet.z_slices, obdet.k_z_integrated, deg=detuning_fit_order)
    alpha_N = p[::-1] # Here I fit the strength
else:
    alpha_N = []

# Build matrix
MM_obj = CouplingMatrix(z_slices, HH, KK, l_min,
        l_max, m_max, n_phi, n_r, N_max, Q_full, sigma_b, r_b,
        a_param, omega0, omega_s, alpha_p=alpha_N,
        pool_size=pool_size)

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


