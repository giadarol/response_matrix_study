import numpy as np
from scipy.constants import c as clight

import PyECLOUD.myfilemanager as mfm

from mode_coupling_matrix import CouplingMatrix

l_min = -3
l_max = 3
m_max = 3
n_phi = 360
n_r = 200
N_max = 20#199

omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
omega_s = 0.001909*omega0
Q_full = 62.31
sigma_b = 1e-9/4*clight
r_b = 4*sigma_b

a_param = 8./r_b**2

Nb_ref = 6e11
ob = mfm.myloadmat_to_obj('../001_sin_response_scan/response_data.mat')
HH = ob.x_mat
KK = ob.dpx_mat
z_slices = ob.z_slices


MM_obj = CouplingMatrix(z_slices, HH, KK, l_min,
        l_max, m_max, n_phi, n_r, N_max, Q_full, sigma_b, r_b,
        a_param)

# Mode coupling test
Nb_array = np.arange(0, 10.5e11, 0.1e11)
Omega_mat = MM_obj.compute_mode_complex_freq(omega_s, rescale_by=Nb_array/Nb_ref)

import matplotlib.pyplot as plt
plt.close('all')
from matplotlib import rc
rc('font', size=14)

mask_unstable = np.imag(Omega_mat) > 0.1
Omega_mat_unstable = Omega_mat.copy()
Omega_mat_unstable[~mask_unstable] = np.nan +1j*np.nan

i_mode = -1

fig1 = plt.figure(200)
plt.plot(Nb_array, np.real(Omega_mat)/omega_s, '.b')
plt.plot(Nb_array, np.real(Omega_mat_unstable)/omega_s, '.r')
plt.grid(True, linestyle=':', alpha=.8)
plt.subplots_adjust(bottom=.12)
plt.suptitle('Response matrix')
plt.xlabel('Bunch intensity [p]')
plt.ylabel(r'Re($\Omega$)/$\omega_s$')
fig1.savefig('response_real.png', dpi=200)


fig2 = plt.figure(201)
plt.plot(Nb_array, np.imag(Omega_mat), '.b')
plt.plot(Nb_array, np.imag(Omega_mat_unstable), '.r')
plt.grid(True, linestyle=':', alpha=.8)
plt.subplots_adjust(bottom=.12)
plt.suptitle('Response matrix')
plt.xlabel('Bunch intensity [p]')
plt.ylabel(r'Im($\Omega$)')
fig2.savefig('response_imag.png', dpi=200)

plt.figure(400)
for ii in range(len(Nb_array)):
    plt.scatter(x=Nb_array[ii]+0*np.imag(Omega_mat[ii, :]),
            y=np.imag(Omega_mat[ii, :]),
            c = np.real(Omega_mat[ii, :])/omega_s,
            cmap=plt.cm.seismic)
plt.colorbar()


# Check against DELPHI
obdelphi = mfm.myloadmat_to_obj('./matrix_delphi.mat')
i_delphi = np.where(np.abs(obdelphi.Nb_vect - Nb_ref)/Nb_ref<1e-3)[0]
kimp_delphi = obdelphi.kimp_vect[i_delphi]
MM_delphi = obdelphi.MM * kimp_delphi

MM = MM_obj.MM
n_l = len(MM_obj.l_vect)
m=0; mp=1;
ratio = [np.mean(np.real(MM[l,m,:,mp])/np.real(MM_delphi[l,m,:,mp]))
            for l in range(n_l)]
l=3;
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(2,1,1)
ax2 = fig1.add_subplot(2,1,2)
ax1.plot(np.real(MM[l,m,:,mp]))
ax1.plot(np.real(MM_delphi[l,m,:,mp]))
ax2.plot(np.real(MM[l,m,:,mp])/np.real(MM_delphi[l,m,:,mp]))
ax2.set_ylim(bottom=0.)

fig2 = plt.figure(2)
plt.plot(MM_obj.l_vect, np.log(ratio)/MM_obj.l_vect, '.')
plt.show()


