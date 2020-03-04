import os

import numpy as np
import matplotlib.pyplot as plt

import PyECLOUD.myfilemanager as mfm

n_terms_to_be_kept = 12
n_tail_cut = 10

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


WW = np.dot(MM, np.dot(RR_inv, FF.T))
WW_filtered = np.dot(MM, np.dot(CC, np.dot(RR_inv, np.dot(FF.T, CC_tails))))


plt.close('all')


fig30 = plt.figure(30)
ax31 = fig30.add_subplot(111)
ax31.matshow(np.abs(WW))

fig31 = plt.figure(31)
ax311 = fig31.add_subplot(111)
ax311.matshow(WW - np.diag(np.diag(WW)))

fig40 = plt.figure(40)
ax41 = fig40.add_subplot(111)
ax41.matshow(np.abs(WW_filtered))

fig50 = plt.figure(50)
ax51 = fig50.add_subplot(111)
import PyECLOUD.mystyle as ms
for ii in range(0, 200, 1):
    colorcurr = ms.colorprog(ii, 200)
    ax51.plot(z_resp, WW[:, ii], color=colorcurr)

fig60 = plt.figure(60)
ax61 = fig60.add_subplot(111)
for ii in range(0, 200, 1):
    colorcurr = ms.colorprog(ii, 200)
    ax61.plot(z_resp-z_resp[ii], WW[:, ii], '-', color=colorcurr)

fig70 = plt.figure(70)
ax71 = fig70.add_subplot(111)
ax71.matshow(np.abs(MM))

plt.show()
