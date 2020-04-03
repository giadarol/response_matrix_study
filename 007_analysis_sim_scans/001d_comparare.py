import numpy as np
import matplotlib.pyplot as plt

import PyECLOUD.myfilemanager as mfm

beta_func = 92.7
T_rev = 88.9e-6
q_frac = .27
Qs = 4.9e-3
l_min = -5
l_max = 3
alpha_0 = -1.61237838e-03
min_strength = 0
max_strength = 1.55
tau_min = 0
tau_max = 300

factor_DQ0 = 0.85
DQ_0 = -alpha_0 * beta_func/4/np.pi*factor_DQ0

dict_plot = {
        't1':  {'fname':'./processed_data/compact_t1_fit.mat', 'tilt_lines':False, 'scale_x':1},
        't2': {'fname':'./processed_data/compact_t2_fit.mat', 'tilt_lines':False, 'scale_x':1},
        #'t2a':{'fname':'./processed_data/compact_t2a_fit.mat', 'tilt_lines':True, 'scale_x':1},
        't2af':{'fname':'./processed_data/compact_t2af_fit.mat', 'tilt_lines':True, 'scale_x':1},
        #'t2b':{'fname':'./processed_data/compact_t2b_fit.mat', 'tilt_lines':True, 'scale_x':1},
        #'t2c':{'fname':'./processed_data/compact_t2c_fit.mat', 'tilt_lines':True, 'scale_x':1},
        't3': {'fname':'./processed_data/compact_t3_fit.mat', 'tilt_lines':True, 'scale_x':1},
        #'t3z': {'fname':'./processed_data/compact_t3z_fit.mat', 'tilt_lines':True, 'scale_x':1},
        #'t3a': {'fname':'./processed_data/compact_t3a_fit.mat', 'tilt_lines':True, 'scale_x':1},
        #'t3b': {'fname':'./processed_data/compact_t3b_fit.mat', 'tilt_lines':True, 'scale_x':1},
        't4': {'fname':'./processed_data/compact_t4_fit.mat', 'tilt_lines':True, 'scale_x':1},
        #'t4c':{'fname':'./processed_data/compact_t4c_fit.mat', 'tilt_lines':True, 'scale_x':1},
        #'t5a':{'fname':'./processed_data/compact_t5a_fit.mat', 'tilt_lines':True, 'scale_x':1},
        'pic':{'fname':'./processed_data/compact_pic_fit.mat', 'tilt_lines':True, 'scale_x':1.},
        }
colorlist = ['b', 'r', 'g']
colorlist = None #['b', 'r', 'g']

plt.close('all')
fig1 = plt.figure(1, figsize=(6.4*1.2, 4.8))
ax1 = fig1.add_subplot(111)
axshare = None
fig_harm_list = []
for ii, ll in enumerate(dict_plot.keys()):
    oo = mfm.myloadmat_to_obj(dict_plot[ll]['fname'])
    tilt_lines = dict_plot[ll]['tilt_lines']
    scale_x = dict_plot[ll]['scale_x']
    kwargs = {}
    if colorlist is not None:
        kwargs['color'] = colorlist[ii]
    ax1.plot(oo.strength_list*scale_x, oo.p_list_centroid/T_rev, label=ll,
            linewidth=2, **kwargs)

    ap_list = oo.ap_list
    N_lines = ap_list.shape[1]
    strength_list = oo.strength_list*scale_x
    freq_list = oo.freq_list
    figharm = plt.figure(100+ii)
    maxsize = np.max(np.array(ap_list))
    axharm = figharm.add_subplot(111, sharex=axshare, sharey=axshare)
    axshare = axharm
    str_mat = np.dot(np.atleast_2d(np.ones(N_lines)).T,
            np.atleast_2d(np.array(strength_list)))
    for lll in range(l_min-10, l_max+10):
        axharm.plot(strength_list, lll + float(tilt_lines)*DQ_0*strength_list/Qs,
                alpha=0.5, linestyle='-', color='grey')
    axharm.scatter(x=str_mat.flatten(),
            y=(np.abs(np.array(freq_list)).T.flatten()-q_frac)/Qs,
            s=np.clip(np.array(ap_list).T.flatten()/maxsize*10, 0.0, 10)*5)
    axharm.set_ylim(l_min, l_max)
    axharm.set_xlim(min_strength, max_strength)
    figharm.suptitle(ll)
    figharm.subplots_adjust(right=.83)
    fig_harm_list.append(figharm)

ax1.legend(bbox_to_anchor=(1, 1),  loc='upper left', fontsize='small')
ax1.grid(True, linestyle=':')
ax1.set_xlim(min_strength, max_strength)
ax1.set_ylim(tau_min, tau_max)
ax1.set_xlabel('e-cloud strength')
ax1.set_ylabel('Instability growth rate [1/s]')
fig1.subplots_adjust(right=.77)
plt.show()
