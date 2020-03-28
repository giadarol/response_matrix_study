import numpy as np
import matplotlib.pyplot as plt

import PyECLOUD.myfilemanager as mfm

dict_plot = {
        't1': './plots_detailed_comparison/compact_t1_fit.mat',
        't2': './plots_detailed_comparison/compact_t2_fit.mat',
        't2a': './plots_detailed_comparison/compact_t2_fit.mat',
        't3': './plots_detailed_comparison/compact_t3_fit.mat',
        't4': './plots_detailed_comparison/compact_t4_fit.mat',
        'pic': './plots_detailed_comparison/compact_pic_fit.mat'}

plt.close('all')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
for ll in dict_plot.keys():
    oo = mfm.myloadmat_to_obj(dict_plot[ll])
