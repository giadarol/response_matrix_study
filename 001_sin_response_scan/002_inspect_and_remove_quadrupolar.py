import os

import numpy as np
import matplotlib.pyplot as plt

import PyECLOUD.myfilemanager as mfm
import PyECLOUD.mystyle as ms
from scipy.signal import savgol_filter

n_terms_to_be_kept = 200
n_tail_cut = 20

# Load response data
response_data_file = 'response_data.mat'
ob_responses = mfm.myloadmat_to_obj(response_data_file)
z_resp = ob_responses.z_slices
x_resp_mat = ob_responses.x_mat
x_resp_mat[np.isnan(x_resp_mat)] = 0.
dpx_resp_mat = ob_responses.dpx_mat
dpx_resp_mat[np.isnan(dpx_resp_mat)] = 0.

# Combine all matrices together
FF = x_resp_mat[:, :].T
MM = dpx_resp_mat[:, :].T
RR = np.dot(FF.T, FF)
RR_inv = np.linalg.inv(RR)

CC = 0*MM
for ii in range(n_terms_to_be_kept):
    CC[ii, ii] = 1

CC_tails = np.identity(len(z_resp))
for ii in range(n_tail_cut):
    CC_tails[ii, ii] = 0.
    CC_tails[-ii, -ii] = 0.


WW = np.dot(MM, np.dot(CC, np.dot(RR_inv, np.dot(FF.T, CC_tails))))
WW_cleaned = np.zeros_like(WW)
n_zero = 30
for i_work in range(len(z_resp)):
    if i_work < n_zero or i_work > len(z_resp) - n_zero:
        continue

    resp_work = WW[:, i_work]

    n_copy = min([len(z_resp)-i_work-2, i_work-2])
    resp_symm = resp_work.copy()
    resp_symm[i_work+2 : i_work+2 + n_copy] = resp_symm[i_work-n_copy-2:i_work-2][::-1]
    resp_symm[i_work-2:i_work+2] = resp_symm[i_work+2]
    resp_filt = savgol_filter(resp_symm, window_length=21, polyorder=5)
    resp_out = resp_filt.copy()
    resp_out[i_work:] = 0.

    WW_cleaned[:, i_work] = resp_out

MM_cleaned = np.dot(WW_cleaned, FF)
WW_check = np.dot(MM_cleaned, np.dot(CC, np.dot(RR_inv, np.dot(FF.T, CC_tails))))
dpx_save = MM_cleaned.T

import scipy.io as sio
sio.savemat('response_data_processed.mat', {
    'dpx_mat': dpx_save,
    'x_mat': x_resp_mat,
    'z_slices': z_resp,
    'sin_ampl_list': ob_responses.sin_ampl_list,
    'cos_ampl_list': ob_responses.cos_ampl_list,
    'n_osc_list': ob_responses.n_osc_list,
    })

plt.close('all')
ms.mystyle_arial(fontsz=14, dist_tick_lab=5, traditional_look=False)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.plot(z_resp, resp_work)
ax1.plot(z_resp, resp_symm)
ax1.plot(z_resp, resp_filt)
ax1.plot(z_resp, resp_out)

WW_plot = WW_check
MM_plot = MM_cleaned

# WW_plot = WW
# MM_plot = MM

fig30 = plt.figure(30)
ax31 = fig30.add_subplot(111)
ax31.matshow(np.abs(WW_plot))

fig31 = plt.figure(31)
ax311 = fig31.add_subplot(111)
ax311.matshow(WW_plot - np.diag(np.diag(WW_plot)))


fig50 = plt.figure(50)
ax51 = fig50.add_subplot(111)
fig60 = plt.figure(60)
ax61 = fig60.add_subplot(111)
import PyECLOUD.mystyle as ms
for ii in list(range(0, 200, 1))[::-2]:
    colorcurr = ms.colorprog(ii, 200)
    ax51.plot(z_resp, WW_plot[:, ii], color=colorcurr)
    ax61.plot(z_resp-z_resp[ii], WW_plot[:, ii], '-', color=colorcurr)

fig70 = plt.figure(70)
ax71 = fig70.add_subplot(111)
ax71.matshow(np.abs(MM_plot))

# Plot frequency responde
fig80 = plt.figure(80)
ax81 = fig80.add_subplot(111)
response_ene = np.sum(MM_plot**2, axis=0)
response_ene = response_ene/np.max(response_ene)
ax81.plot(response_ene[0::2], '.-',
        label='cosine terms', alpha=.6, markeredgewidth=0,
        color='C0', lw=3, markersize=10)
ax81.plot([np.nan]+list(response_ene)[1::2], '.-',
        label='sine terms', alpha=.6, markeredgewidth=0,
        color='C1', lw=3, markersize=10)
ax81.set_ylim(bottom=0)
ax81.set_xlim(left=0)
#ax81.grid(True, linestyle=':')
ax81.legend(loc='upper right', fontsize='medium', frameon=False)
ax81.set_xlabel(r'$f_n^z L_\mathrm{bucket}$')
ax81.set_ylabel(r'$\int k^2_n(z)\,dz$')
fig80.subplots_adjust(bottom=.13, right=.85)
#ax81.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')


plt.show()
