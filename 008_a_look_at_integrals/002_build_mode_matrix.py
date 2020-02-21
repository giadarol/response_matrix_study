import numpy as np

import PyECLOUD.myfilemanager as mfm

lmax = 2
m_max = 2
n_phi = 360
n_r = 200
N_max = 6

ob = mfm.myloadmat_to_obj('../001_sin_response_scan/response_data.mat')

HH = ob.x_mat
KK = ob.dpx_mat

KK[np.isnan(KK)] = 0

