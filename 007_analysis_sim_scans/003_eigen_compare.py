import pickle

import numpy as np
from scipy.constants import c as clight
import scipy.io as sio

import PyECLOUD.mystyle as ms
import PyECLOUD.myfilemanager as mfm



# Plot settings
l_min_plot = -5
l_max_plot = 3
min_strength_plot = 0
max_strength_plot = 1.55
tau_min_plot = 0
tau_max_plot = 300
flag_tilt_lines = False
flag_mode0 = True
fig_fname = 'vlas'

colorlist = None

dict_plot = {
        'onlydip':{'fname':'../008_eigenvalues/eigenvalues.mat'},
        'wquad':{'fname':'../008a1_scan_strength_wlampldet/eigenvalues.mat'}
        }

import matplotlib
import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1, figsize=(6.4*1.2, 4.8))
ax1 = fig1.add_subplot(111)
ms.mystyle_arial(fontsz=14, traditional_look=False)
fig_re_list = []
fig_im_list = []
for ill, ll in enumerate(dict_plot.keys()):

    fname = dict_plot[ll]['fname']

    ob = mfm.myloadmat_to_obj(fname)
    Omega_mat = ob.Omega_mat
    strength_scan = ob.strength_scan
    DQ_0_vect = ob.DQ_0_vect
    M00_array = ob.M00_array
    omega0 = ob.omega0
    omega_s = ob.omega_s
    l_min = ob.l_min
    l_max = ob.l_max
    m_max = ob.m_max
    N_max = ob.N_max

    fig500 = plt.figure(500+ill)#, figsize=(1.3*6.4, 1.3*4.8))
    ax = fig500.add_subplot(111)
    ax.set_facecolor('grey')
    im_min_col = 5
    im_max_col = 200
    im_min_size = 5
    im_max_size = 50
    if flag_mode0:
        maskmode0 = np.abs(np.real(M00_array)/omega_s)<0.9
        ax.plot(strength_scan[maskmode0], np.real(M00_array)[maskmode0]/omega_s, '--',
            linewidth=2, color='w', alpha=0.7)
    for ii in range(len(strength_scan)):
        Omega_ii = Omega_mat[ii, :]
        ind_sorted = np.argsort(-np.imag(Omega_ii))
        re_sorted = np.take(np.real(Omega_ii), ind_sorted)
        im_sorted = np.take(np.imag(Omega_ii), ind_sorted)
        plt.scatter(x=strength_scan[ii]+0*np.imag(Omega_mat[ii, :]),
                y=re_sorted/omega_s,
                c = np.clip(-im_sorted, im_min_col, im_max_col),
                cmap=plt.cm.jet,
                s=np.clip(-im_sorted, im_min_size, im_max_size),
                vmin=im_min_col, vmax=im_max_col,
                norm=matplotlib.colors.LogNorm(),
                )
    plt.suptitle(ll)
    ax.set_xlim(min_strength_plot, max_strength_plot)
    ax.set_ylim(l_min_plot, l_max_plot)
    fig500.subplots_adjust(right=1.)
    for lll in range(l_min_plot-10, l_max_plot+10):
        if flag_tilt_lines:
            add_to_line = np.array(DQ_0_list)*omega0/omega_s
        else:
            add_to_line = 0.
        ax.plot(strength_scan,
                0*strength_scan + lll + add_to_line,
                color='w',
                alpha=.5, linestyle='--')
    ax.tick_params(right=True, top=True)
    plt.colorbar()

    figtau = plt.figure(600+ill)
    axtau = figtau.add_subplot(111)
    axtau.plot(strength_scan, np.imag(Omega_mat),
            '.', color='grey', alpha=0.5)
    axtau.plot(strength_scan, np.max(-np.imag(Omega_mat), axis=1),
            linewidth=2, color='b')
    axtau.set_xlim(min_strength_plot, max_strength_plot)
    axtau.set_ylim(tau_min_plot, tau_max_plot)
    axtau.grid(True, linestyle=':')
    figtau.suptitle(ll)

    kwargs = {}
    if colorlist is not None:
        kwargs['color'] = colorlist[ii]
    ax1.plot(strength_scan, np.max(-np.imag(Omega_mat), axis=1), label=ll,
            linewidth=3, **kwargs)

    fig_re_list.append(fig500)
    fig_im_list.append(figtau)

    fig500.savefig(fig_fname+f'_{ll}_re.png', dpi=200)
    figtau.savefig(fig_fname+f'_{ll}_im.png', dpi=200)

ax1.legend(bbox_to_anchor=(1, 1),  loc='upper left', fontsize='small')
ax1.grid(True, linestyle=':')
#ax1.set_xlim(min_strength, max_strength)
#ax1.set_ylim(tau_min, tau_max)
ax1.set_xlabel('e-cloud strength')
ax1.set_ylabel('Instability growth rate [1/s]')
fig1.subplots_adjust(right=.77)
fig1.savefig(fig_fname + '_glob.png', dpi=200)
plt.show()

