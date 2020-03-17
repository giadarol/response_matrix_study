import pickle

import numpy as np
from scipy.constants import c as clight

import PyECLOUD.mystyle as ms

pkl_fname = 'mode_coupling_matrix.pkl'

l_min = -4 #5
l_max = 4 #5
m_max = 4 #5
N_max = 30
min_imag_unstab = 1.
rescale_to_beta_fun = 92.7

with open(pkl_fname, 'rb') as fid:
    MM_orig = pickle.load(fid)

MM_obj = MM_orig.get_sub_matrix(l_min, l_max, m_max, N_max)

omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
omega_s = 4.9e-3*omega0

# Mode coupling test
strength_scan = np.arange(0, 2.75, 0.02)
Omega_mat = MM_obj.compute_mode_complex_freq(omega_s, rescale_by=strength_scan)

import matplotlib.pyplot as plt
plt.close('all')
ms.mystyle(fontsz=14, traditional_look=False)

mask_unstable = np.imag(Omega_mat) > min_imag_unstab
Omega_mat_unstable = Omega_mat.copy()
Omega_mat_unstable[~mask_unstable] = np.nan+1j*np.nan

i_mode = -1
mask_mode = np.abs(np.real(Omega_mat)/omega_s - i_mode)<0.5
plt.plot(strength_scan, np.imag(Omega_mat), '.b')
plt.plot(strength_scan, np.imag(Omega_mat), '.b')
Omega_mat_mode = Omega_mat.copy()
Omega_mat_mode[~mask_mode] = np.nan

title = f'l_min={l_min}, l_max={l_max}, m_max={m_max}, N_max={N_max}, beta_x={rescale_to_beta_fun}'

plt.figure(200)
plt.plot(MM_orig.beta_fun/rescale_to_beta_fun*strength_scan, np.real(Omega_mat)/omega_s, '.b')
plt.plot(MM_orig.beta_fun/rescale_to_beta_fun*strength_scan, np.real(Omega_mat_unstable)/omega_s, '.r')
plt.suptitle(title)

plt.figure(201)
plt.plot(MM_orig.beta_fun/rescale_to_beta_fun*strength_scan, np.imag(Omega_mat), '.b')
plt.plot(MM_orig.beta_fun/rescale_to_beta_fun*strength_scan, np.imag(Omega_mat_mode), '.g')
plt.suptitle(title)

plt.figure(300)
plt.plot(np.imag(Omega_mat).flatten(),
        np.real(Omega_mat).flatten()/omega_s, '.b')
plt.suptitle(title)


plt.figure(400)
for ii in range(len(strength_scan)):
    plt.scatter(x=MM_orig.beta_fun/rescale_to_beta_fun*strength_scan[ii]+0*np.imag(Omega_mat[ii, :]),
            y=np.imag(Omega_mat[ii, :]),
            c = np.real(Omega_mat[ii, :])/omega_s,
            cmap=plt.cm.seismic)
plt.suptitle(title)
plt.colorbar()

fig500 = plt.figure(500, figsize=(1.3*6.4, 1.3*4.8))
ax = fig500.add_subplot(111)
ax.set_facecolor('grey')
im_min_col = 5
im_max_col = 200
im_min_size = 5
im_max_size = 50
import matplotlib
for ii in range(len(strength_scan)):
    Omega_ii = Omega_mat[ii, :]
    ind_sorted = np.argsort(-np.imag(Omega_ii))
    re_sorted = np.take(np.real(Omega_ii), ind_sorted)
    im_sorted = np.take(np.imag(Omega_ii), ind_sorted)
    plt.scatter(x=MM_orig.beta_fun/rescale_to_beta_fun*strength_scan[ii]+0*np.imag(Omega_mat[ii, :]),
            y=re_sorted/omega_s,
            c = np.clip(-im_sorted, im_min_col, im_max_col),
            cmap=plt.cm.jet,
            s=np.clip(-im_sorted, im_min_size, im_max_size),
            vmin=im_min_col, vmax=im_max_col,
            norm=matplotlib.colors.LogNorm())
plt.suptitle(title)
plt.colorbar()

plt.show()

