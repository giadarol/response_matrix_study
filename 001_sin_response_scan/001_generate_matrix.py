import PyECLOUD.myfilemanager as mfm

import numpy as np


N_samples = 200
ref_ampl = 1e-4
assert(N_samples % 2 ==0)

cos_ampl_list = []
sin_ampl_list = []
n_osc_list = []

for ii in range(N_samples//2):
    cos_ampl_list.append(ref_ampl)
    sin_ampl_list.append(0.)
    n_osc_list.append(ii)

    cos_ampl_list.append(0.)
    sin_ampl_list.append(ref_ampl)
    n_osc_list.append(ii+1)

x_meas_mat = []
x_mat = []
dpx_mat = []
for ii in range(len(n_osc_list)):

    cos_ampl = cos_ampl_list[ii]
    sin_ampl = sin_ampl_list[ii]
    n_osc = n_osc_list[ii]

    current_sim_ident= f'n_{n_osc:.1f}_c{cos_ampl:.2e}_s{sin_ampl:.2e}'
    ob = mfm.myloadmat_to_obj('../simulations/' + current_sim_ident + '/response.mat')

    x_mat.append(ob.x_ideal)
    x_meas_mat.append(ob.x_slices)
    dpx_mat.append(ob.dpx_slices)

z_slices = ob.z_slices

import scipy.io as sio
sio.savemat('response_data.mat',{
    'x_mat': x_mat,
    'z_slices': z_slices,
    'dpx_mat': dpx_mat})


x_mat = np.array(x_mat)

x_mat[np.isnan(x_mat)] = 0.

f_mat = x_mat

N_base = f_mat.shape[0]

w_mat = f_mat

M_mat = np.dot(f_mat, w_mat.T)

x_test = 5e-3 * z_slices

b_test = np.dot(w_mat, x_test.T)

a_test = np.linalg.solve(M_mat, b_test)

x_check = np.dot(a_test, f_mat)
b_check = np.dot(f_mat, x_check.T)
