import numpy as np
from scipy.special import assoc_laguerre, gamma
from scipy.constants import c as clight
from numpy.linalg import eigvals

import PyECLOUD.myfilemanager as mfm

l_min = -3
l_max = 3
m_max = 3
n_phi = 360
n_r = 200
N_max = 199

omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
omega_s = 0.001909*omega0
Q_full = 62.31
sigma_b = 1e-9/4*clight
r_b = 4*sigma_b

a_param = 8./r_b**2

ob = mfm.myloadmat_to_obj('../001_sin_response_scan/response_data.mat')

r_max = np.max(np.abs(ob.z_slices))
dz = ob.z_slices[1] - ob.z_slices[0]

r_vect = np.linspace(0, r_max, n_r)
phi_vect = np.linspace(0, 2*np.pi, n_phi)

dphi = phi_vect[1] - phi_vect[0]
dr = r_vect[1] - r_vect[0]

l_vect = np.array(range(l_min, l_max+1))
m_vect = np.array(range(0, m_max+1))

n_l = len(l_vect)
n_m = len(m_vect)
n_n = N_max + 1

HH = ob.x_mat
KK = ob.dpx_mat
z_slices = ob.z_slices

KK[np.isnan(KK)] = 0

H_N_2_vect = dz * np.sum(HH**2, axis=1)

cos_phi = np.cos(phi_vect)
cos2_phi = cos_phi*cos_phi

e_L_PHI_mat = np.zeros((n_l, n_phi), dtype=np.complex)
for i_l, ll in enumerate(l_vect):
    e_L_PHI_mat[i_l, :] = np.exp(1j*ll*phi_vect)

# Remember that Ks and Hs do not have the last point at 360 deg
# Compute R_tilde integrals
print('Compute R_tilde_lmn ...')
R_tilde_lmn = np.zeros((n_l, n_m, n_n), dtype=np.complex)
for i_l, ll in enumerate(l_vect):
    print(f'{i_l}/{n_l}')
    r_part_l_M_R_mat = np.zeros((n_m, n_r))
    for i_m, mm in  enumerate(m_vect):
        lag_l_m_R_vect =assoc_laguerre(
                a_param * r_vect*r_vect, n=mm, k=np.abs(ll))
        r_part_l_M_R_mat[i_m, :]  = (
                  dr * r_vect
                * (r_vect/r_b)**np.abs(ll)
                * lag_l_m_R_vect
                )
        for nn in range(n_n):
            int_dphi_l_n_R_vect = np.zeros(n_r, dtype=np.complex)
            for i_r, rr in enumerate(r_vect):
                h_n_r_cos_phi = np.interp(rr*cos_phi,
                    z_slices, HH[nn, :])
                exp_c2_r_PHI_vect = np.exp(-a_param*rr*rr
                        *(1-cos2_phi/(2*a_param*sigma_b**2)))
                int_dphi_l_n_R_vect[i_r] = dphi * np.sum(
                    exp_c2_r_PHI_vect
                  * h_n_r_cos_phi/H_N_2_vect[nn]
                  * np.conj(e_L_PHI_mat[i_l, :]))

            R_tilde_lmn[i_l, i_m, nn] = np.sum(
                    r_part_l_M_R_mat[i_m, :]*
                    int_dphi_l_n_R_vect)

# Compute R integrals
print('Compute R_lmn ...')
R_lmn = np.zeros((n_l, n_m, n_n), dtype=np.complex)
for i_l, ll in enumerate(l_vect):
    print(f'{i_l}/{n_l}')
    r_part_l_M_R_mat = np.zeros((n_m, n_r))
    for i_m, mm in  enumerate(m_vect):
        lag_l_m_R_vect =assoc_laguerre(
                a_param * r_vect*r_vect, n=mm, k=np.abs(ll))
        r_part_l_M_R_mat[i_m, :]  = (
                  dr * r_vect
                * (a_param*r_b*r_vect)**np.abs(ll)
                * lag_l_m_R_vect
                * np.exp(-r_vect**2 / (2*sigma_b**2))
                )

        for nn in range(n_n):
            int_dphi_l_n_R_vect = np.zeros(n_r, dtype=np.complex)
            for i_r, rr in enumerate(r_vect):
                k_n_r_cos_phi = np.interp(rr*cos_phi,
                    z_slices, KK[nn, :])
                int_dphi_l_n_R_vect[i_r] = dphi * np.sum(
                    k_n_r_cos_phi*e_L_PHI_mat[i_l, :])

            R_lmn[i_l, i_m, nn] = np.sum(
                    r_part_l_M_R_mat[i_m, :]*
                    int_dphi_l_n_R_vect)

print('Compute final matrix')
no_coeff_M_l_m_lp_mp = np.zeros((n_l, n_m, n_l, n_m), dtype=np.complex)
for i_l, ll in enumerate(l_vect):
    for i_m, mm in enumerate(m_vect):
        for i_lp in range(n_l):
            for i_mp in range(n_m):
                temp = gamma(mm + 1) / gamma(np.abs(ll) + mm + 1)

#                # To be handled:
#                assert(not(np.isnan(temp)))
#                assert(not(np.isinf(temp)))

                no_coeff_M_l_m_lp_mp[i_l, i_m, i_lp, i_mp] = (
                        temp * np.sum(R_tilde_lmn[i_lp, i_mp, :]
                            * R_lmn[i_l, i_m, :]))

coeff = -clight*a_param/(4*np.pi**2*np.sqrt(2*np.pi)*Q_full*sigma_b)
MM = coeff*no_coeff_M_l_m_lp_mp


obdelphi = mfm.myloadmat_to_obj('./matrix_delphi.mat')
MM_delphi = obdelphi.MM *obdelphi.kimp

# Mode coupling test
Nb_ref = 6e11
Nb_array = np.arange(0, 10.5e11, 1e11)
Omega_mat = []
for ii, Nb in enumerate(Nb_array):

    MM_m_l_omegas = MM_delphi.copy()
    MM_m_l_omegas *= (Nb/Nb_ref)

    for i_l, ll in enumerate(l_vect):
        for i_m, mm in enumerate(m_vect):
            for i_lp in range(n_l):
                for i_mp in range(n_m):
                    if i_l == i_lp:
                        MM_m_l_omegas += ll*omega_s
    # Check against DELPHI
    mat_to_diag = MM_m_l_omegas.reshape((n_l*n_m,n_l*n_m))
    Omegas=eigvals(mat_to_diag)
    Omega_mat.append(Omegas)

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(200)
plt.plot(Nb_array, np.real(Omega_mat)/omega_s, '.b')

plt.figure(201)
plt.plot(Nb_array, np.imag(Omega_mat)/omega_s, '.b')


m=0; mp=1;
ratio = [np.mean(np.real(MM[l,m,:,mp])/np.real(MM_delphi[l,m,:,mp])) for l in range(n_l)]
l=3;
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(2,1,1)
ax2 = fig1.add_subplot(2,1,2)
ax1.plot(np.real(MM[l,m,:,mp]))
ax1.plot(np.real(MM_delphi[l,m,:,mp]))
ax2.plot(np.real(MM[l,m,:,mp])/np.real(MM_delphi[l,m,:,mp]))
ax2.set_ylim(bottom=0.)

fig2 = plt.figure(2)
plt.plot(l_vect, np.log(ratio)/l_vect, '.')
plt.show()


