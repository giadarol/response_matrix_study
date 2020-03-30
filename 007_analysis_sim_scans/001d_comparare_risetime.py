import numpy as np
import matplotlib.pyplot as plt

import PyECLOUD.myfilemanager as mfm

dict_plot = {
        #'t1':  './processed_data/compact_t1_fit.mat',
        't2':  './processed_data/compact_t2_fit.mat',
        #'t2a': './processed_data/compact_t2a_fit.mat',
        #'t2b': './processed_data/compact_t2b_fit.mat',
        't2c': './processed_data/compact_t2c_fit.mat',
        #'t3':  './processed_data/compact_t3_fit.mat',
        't4':  './processed_data/compact_t4_fit.mat',
        't4c':  './processed_data/compact_t4c_fit.mat',
        'pic': './processed_data/compact_pic_fit.mat'
        }

plt.close('all')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
for ll in dict_plot.keys():
    oo = mfm.myloadmat_to_obj(dict_plot[ll])
    ax1.plot(oo.strength_list, oo.p_list_centroid, label=ll)

ax1.legend(fontsize='small')

plt.show()
