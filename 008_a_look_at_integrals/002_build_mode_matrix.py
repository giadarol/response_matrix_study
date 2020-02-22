import numpy as np
from scipy.special import assoc_laguerre
from scipy.constants import c as clight

import PyECLOUD.myfilemanager as mfm

l_min = -1
l_max = 3
m_min = -2
m_max = 3
n_phi = 360
n_r = 200
N_max = 6

sigma_b = 1e-9/4*clight
r_b = 4*sigma_b

a_param = 8/r_b**2

ob = mfm.myloadmat_to_obj('../001_sin_response_scan/response_data.mat')

r_max = np.max(np.abs(ob.z_slices))

r_vect = np.linspace(0, r_max, n_r)
phi_vect = np.linspace(0, 2*np.pi, n_phi)

dphi = phi_vect[1] - phi_vect[0]
dr = r_vect[1] - r_vect[0]

l_vect = np.array(range(l_min, l_max+1))
m_vect = np.array(range(m_min, m_max+1))

n_l = len(l_vect)
n_m = len(m_vect)
n_n = N_max + 1

HH = ob.x_mat
KK = ob.dpx_mat
z_slices = ob.z_slices

KK[np.isnan(KK)] = 0

cos_phi = np.cos(phi_vect)

e_l_phi_mat = np.zeros((n_l, n_phi), dtype=np.complex)
for i_l, ll in enumerate(l_vect):
    e_l_phi_mat[i_l, :] = np.exp(1j*ll*phi_vect)

# Remember that Ks and Hs do not have the last point at 360 deg
R_lmn = np.zeros((n_l, n_m, n_m), dtype=np.complex)
for i_m, mm in enumerate(m_vect):

    r_part_mat = np.zeros((n_l, n_r))
    for i_l, ll in  enumerate(l_vect):
        lag_curr =assoc_laguerre(
                a_param * r_vect*r_vect, n=mm, k=np.abs(ll))
        r_part_mat[i_l, :]  = (
                  dr * r
                * (a_param/r_b*r_vect)**np.abs(l)
                * lag_curr
                * exp(-r_vect**2 / (2*sigma_b**2))

    for i_r, rr in enumerate(r_vect):
        for nn in range(n_n):
            k_n_r_cos_phi = np.interp(rr*cos_phi,
                    z_slices, KK[nn, :])
            for i_l, ll in enumerate(l_vect):
                int_dphi_l_n_r = dphi * np.sum(
                        k_n_r_cos_phi*e_l_phi_mat[i_l, :])


