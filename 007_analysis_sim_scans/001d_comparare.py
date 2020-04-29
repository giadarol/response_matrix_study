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
max_strength = 1.5
tau_min = 0
tau_max = 300

factor_DQ0 = 0.85
DQ_0 = -alpha_0 * beta_func/4/np.pi*factor_DQ0

dict_plot = {
        #'t1':  {'fname':'./processed_data/compact_t1_fit.mat', 'tilt_lines':False, 'scale_x':1},
        #'t2': {'fname':'./processed_data/compact_t2_fit.mat', 'tilt_lines':False, 'scale_x':1},
        #'t2af':{'fname':'./processed_data/compact_t2af_fit.mat', 'tilt_lines':True, 'scale_x':1},
        #'t3': {'fname':'./processed_data/compact_t3_fit.mat', 'tilt_lines':True, 'scale_x':1},
        #'t4': {'fname':'./processed_data/compact_t4_fit.mat', 'tilt_lines':True, 'scale_x':1},
        #'t6': {'fname':'./processed_data/compact_t6_fit.mat', 'tilt_lines':True, 'scale_x':1},
        'pic':{'fname':'./processed_data/compact_pic_fit.mat', 'tilt_lines':True, 'scale_x':1.},
        #'picQp5':{'fname':'./processed_data/compact_picQp5_fit.mat', 'tilt_lines':True, 'scale_x':1.},
        }
colorlist = ['b', 'r', 'g', 'orange', 'k']
colorlist = ['C0', 'C3']
#colorlist = None



def extract_independent_lines(strength_list,
        all_freqs, all_aps, min_dist, n_indep_list,
        allowed_range=None):

    all_freq_indep = []
    all_aps_indep = []
    all_stre_indep = []
    for jjj, sss in enumerate(strength_list):
        this_freqs = np.abs(all_freqs[jjj, :])
        this_aps = np.abs(all_aps[jjj, :])

        i_sorted = np.argsort(this_aps)[::-1]

        this_f_indep = [0]
        this_ap_indep = [0]
        this_stren_indep = [0]
        for ifr in i_sorted:
            if len(this_f_indep) > n_indep_list[jjj]:
                break
            ff = this_freqs[ifr]
            if allowed_range is not None:
                if ff>allowed_range[1] or ff<allowed_range[0]:
                    continue
            if np.min(np.abs(ff - np.array(this_f_indep))) > min_dist:
                this_f_indep.append(ff)
                this_ap_indep.append(this_aps[ifr])
                this_stren_indep.append(sss)

        all_freq_indep += this_f_indep[1:]
        all_aps_indep += this_ap_indep[1:]
        all_stre_indep += this_stren_indep[1:]

    return all_freq_indep, all_aps_indep, all_stre_indep


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

    mask_strength = (oo.strength_list <= max_strength)

    ap_list = oo.ap_list
    N_lines = ap_list.shape[1]
    strength_list = oo.strength_list[mask_strength]*scale_x
    freq_list = oo.freq_list[mask_strength, :]
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
            s=np.clip(np.array(ap_list).T.flatten()/maxsize*10, 0.0, 10))
    axharm.set_ylim(l_min, l_max)
    axharm.set_xlim(min_strength, max_strength)
    figharm.suptitle(ll)
    figharm.subplots_adjust(right=.83)
    fig_harm_list.append(figharm)
    all_freq_indep_0, all_aps_indep_0, all_stre_indep_0 = extract_independent_lines(
        strength_list, np.abs(np.array(freq_list)), np.array(ap_list),
        min_dist=3e-3, n_indep_list=np.zeros_like(strength_list, dtype=np.int)+3)
    indep_normalized_0 = (np.array(all_freq_indep_0)-.27)/Qs
    mask_keep_0 = np.abs(indep_normalized_0)<1.5
    axharm.plot(np.array(all_stre_indep_0)[mask_keep_0],
                indep_normalized_0[mask_keep_0], '.', color='C03')

    freq_mode_0, ap_mode_0, stre_mode_0 = extract_independent_lines(
        strength_list, np.abs(np.array(freq_list)), np.array(ap_list),
        min_dist=3e-3, n_indep_list=np.zeros_like(strength_list, dtype=np.int)+1,
        allowed_range=(.27, .27+4e-3))
    # Plot data from intrabunch motion
    all_freqs = np.concatenate((oo.freqs_1mode_re_list, oo.freqs_1mode_im_list), axis=1)
    all_aps = np.concatenate((oo.ap_1mode_re_list, oo.ap_1mode_im_list), axis=1)
    maxsizeintra = np.max(np.array(all_aps))
    figintra = plt.figure(200+ii)
    axintra = figintra.add_subplot(111, sharex=axshare, sharey=axshare)
    str_mat_intra = np.dot(np.atleast_2d(np.ones(all_freqs.shape[1])).T,
            np.atleast_2d(np.array(strength_list)))
    axintra.scatter(x=str_mat_intra.flatten(),
            y=(np.abs(np.array(all_freqs)).T.flatten()-q_frac)/Qs,
            s=np.clip(np.array(all_aps).T.flatten()/maxsizeintra*10, 0.0, 10))

    min_dist = 3e-3
    n_indep_list = np.zeros_like(strength_list, dtype=np.int) + 50
    n_indep_list[oo.n_sample_list<1000] = 1
    all_freq_indep, all_aps_indep, all_stre_indep = extract_independent_lines(
        strength_list, all_freqs, all_aps, min_dist, n_indep_list)

    indep_normalized = (np.array(all_freq_indep)-.27)/Qs
    mask_keep = np.abs(indep_normalized)<1.5
    axintra.plot(np.array(all_stre_indep)[mask_keep],
                indep_normalized[mask_keep], '.', color='C03')

ax1.legend(bbox_to_anchor=(1, 1),  loc='upper left', fontsize='small')
ax1.grid(True, linestyle=':')
ax1.set_xlim(min_strength, max_strength)
ax1.set_ylim(tau_min, tau_max)
ax1.set_xlabel('e-cloud strength')
ax1.set_ylabel('Instability growth rate [1/s]')
fig1.subplots_adjust(right=.77)
plt.show()
