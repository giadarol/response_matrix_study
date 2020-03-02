import os,sys
sys.path.append("tools")
sys.path.append("PyHEADTAIL")

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob

from scipy.signal import savgol_filter
import scipy.io as sio

from PyPARIS_sim_class import LHC_custom
from PyPARIS_sim_class import propsort as ps

import PyECLOUD.myfilemanager as mfm
import PyECLOUD.mystyle as ms

from scipy.constants import c as ccc

# # Test
# labels = [f'test', 'reference']
# folders_compare = ['../004_instability_simulation', '../reference_simulation/']
# fname = None
# fft2mod = 'lin'
# i_start_list = None
# n_turns = 20*[1000000]
# cmap = None

# Detuning investigation 
labels = [f'full', 'lin', 'nonlin', 'mat_only', 'lin_ave', 'lin2ave',
        'lin1.75ave', 'lin_no_ave']
folders_compare = ['../004a_instability_investigate_detuning_full',
        '../004b_instability_investigate_detuning_limap',
        '../004c_instability_investigate_detuning_nonlinmap',
        '../004d_instability_investigate_detuning_matrix_only',
        '../004e_instability_investigate_detuning_limap_ave',
        '../004f_instability_investigate_detuning_limap_2ave',
        '../004g_instability_investigate_detuning_limap_1.75ave',
        '../004h_instability_investigate_detuning_limap_noave',
        ]
fname = 'compact_comparison'
fft2mod = 'lin'
i_start_list = None
n_turns = 20*[1000000]
cmap = plt.cm.rainbow
i_force_line = None
fit_cut = 2000
flag_compact = True

# # Comparison against full study
# VRF_MV = 8
# labels = ['matrix_map', 'matrix_only', 'map_only', 'simulation']
# folders_compare = [
#     f'../005k_voltage_scan_all_harmonics_damper_off_matrix_map/simulations/V_RF_{VRF_MV:.1e}',
#     f'../005l_voltage_scan_all_harmonics_damper_off_matrix_only/simulations/V_RF_{VRF_MV:.1e}',
#     f'../005c_voltage_scan_map_only/simulations/V_RF_{VRF_MV:.1e}',
#     ('/afs/cern.ch/project/spsecloud/Sim_PyPARIS_017/'
#         'inj_arcQuad_drift_sey_1.4_intensity_1.2e11ppb_sigmaz_97mm_VRF_3_8MV_yes_no_initial_kick/'
#          'simulations_PyPARIS/'
#          f'ArcQuad_no_initial_kick_T0_x_slices_500_segments_8_MPslice_2500_eMPs_5e5_length_07_sey_1.4_intensity_1.2e11ppb_VRF_{VRF_MV:d}MV')
#     ]
# fname = f'comparison_Vrf{VRF_MV:.1f}_MV'
# fft2mod = 'log'
# i_start_list = [3000, 3000, 3000, 2000]
# n_turns = 6*[3500]
# cmap = None
# i_force_line = 2
# fit_cut = 2000
# flag_compact = True

## Comparison with Luca
#V_RF=3
#labels = ['new no recenter', 'new recenter', 'old']
#folders_compare = [
#        f'/afs/cern.ch/project/spsecloud/Sim_PyPARIS_019/inj_arcQuad_no_initial_kick_no_damper_sey_1.4_intensity_1.2e11ppb_sigmaz_97mm_scan_VRF_3_8MV_recenter_slice_yes_no/simulations_PyPARIS/initial_kick_no_damper_no_recenter_slices_False_VRF_{V_RF:.0f}MV',
#    f'/afs/cern.ch/project/spsecloud/Sim_PyPARIS_019/inj_arcQuad_no_initial_kick_no_damper_sey_1.4_intensity_1.2e11ppb_sigmaz_97mm_scan_VRF_3_8MV_recenter_slice_yes_no/simulations_PyPARIS/initial_kick_no_damper_no_recenter_slices_True_VRF_{V_RF:.0f}MV',
#    f'/afs/cern.ch/project/spsecloud/Sim_PyPARIS_017/inj_arcQuad_drift_sey_1.4_intensity_1.2e11ppb_sigmaz_97mm_VRF_3_8MV_yes_no_initial_kick/simulations_PyPARIS/ArcQuad_no_initial_kick_T0_x_slices_500_segments_8_MPslice_2500_eMPs_5e5_length_07_sey_1.4_intensity_1.2e11ppb_VRF_{V_RF:.0f}MV'
#    ]
#fname = None
#fft2mod = 'lin'
#i_start_list = None
#n_turns = 6*[10000]
#cmap = plt.cm.rainbow
#i_force_line = 2
#fit_cut = 2000
#flag_compact = True

# # Comparison v
# V_list = np.arange(3, 8.1, 1)
# labels = [f'{vv:.1f}_MV' for vv in V_list]
# folders_compare = [
# #    f'../005a_voltage_scan_matrix_map/simulations/V_RF_{vv:.1e}' for vv in V_list]
# #    f'../005c_voltage_scan_map_only/simulations/V_RF_{vv:.1e}' for vv in V_list]
# #    f'../005b_voltage_scan_matrix_only/simulations/V_RF_{vv:.1e}' for vv in V_list]
# #    f'../005g_voltage_scan_all_harmonics_matrix_map/simulations/V_RF_{vv:.1e}' for vv in V_list]
#     f'../005k_voltage_scan_all_harmonics_damper_off_matrix_map/simulations/V_RF_{vv:.1e}' for vv in V_list]
# #    f'../005l_voltage_scan_all_harmonics_damper_off_matrix_only/simulations/V_RF_{vv:.1e}' for vv in V_list]
# #      ('/afs/cern.ch/project/spsecloud/Sim_PyPARIS_017/'
# #          'inj_arcQuad_drift_sey_1.4_intensity_1.2e11ppb_sigmaz_97mm_VRF_3_8MV_yes_no_initial_kick/'
# #           'simulations_PyPARIS/'
# #      f'ArcQuad_yes_initial_kick_T0_x_slices_500_segments_8_MPslice_2500_eMPs_5e5_length_07_sey_1.4_intensity_1.2e11ppb_VRF_{vv:d}MV') for vv in np.int_(V_list)]
# fname = 'compact_forced_vscan_matrix_200harm_nodamper_mat_map'
# #fname = 'compact_forced_vscan_sim_kick'
# #fname = None
# fft2mod = 'lin'
# i_start_list = None
# n_turns = 12*[10000000]
# cmap = plt.cm.rainbow
# i_force_line = 2
# fit_cut = 5000
# flag_compact = True

# # Comparison strength
# strength_list = np.arange(0.1, 2.1, 0.1)
# labels = [f'strength {ss:.1f}' for ss in strength_list]
# folders_compare = [
# #     f'../005d_strength_scan_6MV_matrix_map/simulations/strength_{ss:.2e}/' for ss in strength_list]
# #     f'../005e_strength_scan_6MV_matrix_only/simulations/strength_{ss:.2e}/' for ss in strength_list]
# #     f'../005f_strength_scan_6MV_map_only/simulations/strength_{ss:.2e}/' for ss in strength_list]
# #     f'../005i_strength_scan_6MV_all_harmonics_matrix_map/simulations/strength_{ss:.2e}/' for ss in strength_list]
# #     f'../005j_strength_scan_6MV_all_harmonics_matrix_only/simulations/strength_{ss:.2e}/' for ss in strength_list]
#      f'../005m_strength_scan_6MV_all_harmonics_matrix_linmap/simulations/strength_{ss:.2e}/' for ss in strength_list]
# #     ('/afs/cern.ch/project/spsecloud/Sim_PyPARIS_019/'
# #      'inj_arcQuad_no_initial_kick_no_damper_recenter_slice_sey_1.4'
# #      '_intensity_1.2e11ppb_VRF_6MV_scan_length_factor_0.1_2.0/'
# #      'simulations_PyPARIS/'
# #      f'initial_kick_no_damper_no_recenter_slices_length_factor_{ss:.1f}') for ss in strength_list]
# fft2mod = 'lin'
# fname = 'compact_forced_strength_scan_maplin'
# # fname = None
# i_start_list = None
# n_turns = 30*[10000000]
# cmap = plt.cm.rainbow
# i_force_line = 2
# fit_cut = 5000
# flag_compact = True
# #######################################################################

flag_naff = False

def extract_info_from_sim_param(spfname):
    with open(spfname, 'r') as fid:
        lines = fid.readlines()

    ddd = {}
    # Extract V_RF
    for ll in lines:
        if '=' in ll:
            nn = ll.split('=')[0].replace(' ','')
            try:
                ddd[nn] = eval(ll.split('=')[-1])
            except:
                ddd[nn] = 'Failed!'
    return ddd

plt.close('all')

fig1 = plt.figure(1, figsize=(8/1.3,6*1.5/1.3))
ax11 = fig1.add_subplot(3,1,1)
ax12 = fig1.add_subplot(3,1,2, sharex=ax11)
ax13 = fig1.add_subplot(3,1,3, sharex=ax11)

p_list = []
for ifol, folder in enumerate(folders_compare):

    print('Folder %d/%d'%(ifol, len(folders_compare)))

    folder_curr_sim = folder
    sim_curr_list = ps.sort_properly(glob.glob(folder_curr_sim+'/bunch_evolution_*.h5'))
    # sim_curr_list = [folder_curr_sim+'/bunch_evolution.h5']
    ob = mfm.monitorh5list_to_obj(sim_curr_list)

    sim_curr_list_slice_ev = ps.sort_properly(glob.glob(folder_curr_sim+'/slice_evolution_*.h5'))
    # sim_curr_list_slice_ev = [folder_curr_sim+'/slice_evolution.h5']
    ob_slice = mfm.monitorh5list_to_obj(sim_curr_list_slice_ev, key='Slices', flag_transpose=True)

    try:
        import pickle
        with open(folder+'/sim_param.pkl', 'rb') as fid:
            pars = pickle.load(fid)
    except IOError:
        config_module_file = folder+'/Simulation_parameters.py'
        print('Config pickle not found, loading from module:')
        print(config_module_file)
        pars = mfm.obj_from_dict(
                extract_info_from_sim_param(config_module_file))


    w_slices = ob_slice.n_macroparticles_per_slice
    wx = ob_slice.mean_x * w_slices / np.mean(w_slices)
    rms_x = np.sqrt(np.mean((ob_slice.mean_x * w_slices)**2, axis=0))
    mask_zero = ob.epsn_x > 0.
    mask_zero[n_turns[ifol]:] = False

    if cmap is not None:
        cc = cmap(float(ifol)/float(len(folders_compare)))
        kwargs = {'color': cc}
    else:
        kwargs = {}
    ax11.plot(ob.mean_x[mask_zero]*1e3, label=labels[ifol], **kwargs)
    ax12.plot(ob.epsn_x[mask_zero]*1e6, **kwargs)
    intrabunch_activity = savgol_filter(rms_x[mask_zero], 21, 3)
    # ax13.plot(intrabunch_activity, **kwargs)

    import sys
    sys.path.append('./NAFFlib')

    figfft = plt.figure(300)
    axfft = figfft.add_subplot(111)

    figffts = plt.figure(3000 + ifol, figsize=(1.7*6.4, 1.8*4.8))
    plt.rcParams.update({'font.size': 12})

    axwidth = .38
    pos_col1 = 0.1
    pos_col2 = 0.57
    pos_row1 = 0.63
    height_row1 = 0.3
    pos_row2 = 0.37
    height_row2 = 0.18
    pos_row3 = 0.07
    height_row3 = 0.22

    axffts = figffts.add_axes((pos_col2, pos_row1, axwidth, height_row1))
    axfft2 = figffts.add_axes((pos_col1, pos_row1, axwidth, height_row1), sharey=axffts)
    axcentroid = figffts.add_axes((pos_col2, pos_row2, axwidth, height_row2),
        sharex=axffts)
    ax1mode = figffts.add_axes((pos_col1, pos_row2, axwidth, height_row2),
        sharex=axcentroid)
    axtraces = figffts.add_axes((pos_col1, pos_row3, axwidth, height_row3))
    axtext = figffts.add_axes((pos_col2, pos_row3, axwidth, height_row3))

    #axtraces = plt.subplot2grid(fig=figffts, shape=(3,4), loc=(2,1), colspan=2)

    figffts.subplots_adjust(
        top=0.925,
        bottom=0.07,
        left=0.11,
        right=0.95,
        hspace=0.3,
        wspace=0.28)

    fftx = np.fft.rfft(ob.mean_x[mask_zero])
    qax = np.fft.rfftfreq(len(ob.mean_x[mask_zero]))
    axfft.semilogy(qax, np.abs(fftx), label=labels[ifol])

    # I try some NAFF on the centroid
    import NAFFlib as nl
    if flag_naff:

        n_wind = 50
        N_lines = 10
        freq_list = []
        ampl_list = []

        x_vect = ob.mean_x[mask_zero]
        N_samples = len(x_vect)

        for ii in range(N_samples):
            if ii < n_wind/2:
                continue
            if ii > N_samples-n_wind/2:
                continue

            freq, a1, a2 = nl.get_tunes(
                    x_vect[ii-n_wind/2 : ii+n_wind/2], N_lines)
            freq_list.append(freq)
            ampl_list.append(np.abs(a1))

        fignaff = plt.figure(301)
        axnaff = fignaff.add_subplot(111)

        mpbl = axnaff.scatter(x=np.array(N_lines*[np.arange(len(freq_list))]).T,
            y=np.array(freq_list), c=(np.array(ampl_list)),
            vmax=1*np.max(ampl_list),
            s=1)
        plt.colorbar(mpbl)

    # Details
    L_zframe = np.max(ob_slice.mean_z[:, 0]) - np.min(ob_slice.mean_z[:, 0])
    # I try some FFT on the slice motion
    ffts = np.fft.fft(wx, axis=0)
    n_osc_axis = np.arange(ffts.shape[0])*4*ob.sigma_z[0]/L_zframe
    axffts.pcolormesh(np.arange(wx.shape[1]), n_osc_axis, np.abs(ffts))
    axffts.set_ylim(0, 5)
    axffts.set_ylabel('N. oscillations\nin 4 sigmaz')
    axffts.set_xlabel('Turn')

    # I try a double fft
    fft2 = np.fft.fft(ffts, axis=1)
    q_axis_fft2 = np.arange(0, 1., 1./wx.shape[1])
    if fft2mod=='log':
        matplot = np.log(np.abs(fft2))
    else:
        matplot = np.abs(fft2)
    axfft2.pcolormesh(q_axis_fft2,
            n_osc_axis, matplot)
    axfft2.set_ylabel('N. oscillations\nin 4 sigmaz')
    axfft2.set_ylim(0, 5)
    axfft2.set_xlim(0.25, .30)
    axfft2.set_xlabel('Tune')

    axcentroid.plot(ob.mean_x[mask_zero]*1000)
    axcentroid.set_xlabel('Turn')
    axcentroid.set_ylabel('Centroid position [mm]')
    axcentroid.grid(True, linestyle='--', alpha=0.5)
    axcentroid.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    # Plot time evolution of most unstable "mode"
    if i_force_line is None:
        i_mode = np.argmax(
            np.max(np.abs(ffts[:ffts.shape[0]//2, mask_zero][:, :-50]), axis=1)\
          - np.max(np.abs(ffts[:ffts.shape[0]//2, mask_zero][:, :50]), axis=1))
        forced = False
    else:
        i_mode = i_force_line
        forced = True
    ax1mode.plot(np.real(ffts[i_mode, :][mask_zero]), label = 'cos comp.')
    ax1mode.plot(np.imag(ffts[i_mode, :][mask_zero]), alpha=0.5, label='sin comp.')
    ax1mode.legend(loc='upper left', prop={'size':11})
    ax1mode.set_xlabel('Turn')
    ax1mode.set_ylabel(f'Line with {n_osc_axis[i_mode]:.2f} osc.')
    ax1mode.grid(True, linestyle='--', alpha=0.5)
    ax1mode.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    ax1mode.set_xlim(0, np.sum(mask_zero))

    activity_mode = savgol_filter(np.abs(ffts[i_mode, :][mask_zero]), 41, 3)
    x_fit = np.arange(len(activity_mode), dtype=np.float)
    p_fit = np.polyfit(x_fit[20:fit_cut], np.log(activity_mode)[20:fit_cut], deg = 1)
    y_fit = np.polyval(p_fit, x_fit)
    p_list.append(p_fit)
    tau = 1./p_fit[0]

    ax13.plot(np.log(activity_mode), **kwargs)
    ax13.plot(y_fit, **kwargs)

    for ax in [axcentroid, ax1mode]:
        ax.set_ylim(np.array([-1, 1])*np.max(np.abs(np.array(ax.get_ylim()))))

    tune_centroid = nl.get_tune(ob.mean_x[mask_zero])
    tune_1mode_re = nl.get_tune(np.real(ffts[i_mode, :]))
    tune_1mode_im = nl.get_tune(np.imag(ffts[i_mode, :]))

    if flag_compact:
        ax1mode.text(0.02, 0.02,
            (f'Tau: {tau:.0f} turns\n'
             f'Tunes: {tune_1mode_re:.4f}/{tune_1mode_im:.4f}'),
            transform=ax1mode.transAxes,
            ha='left', va='bottom', fontsize=11)

    N_traces =15 
    max_intr = np.max(intrabunch_activity)
    if i_start_list is None:
        try:
            #i_start = np.where(intrabunch_activity<0.3*max_intr)[0][-1] - N_traces
            i_start = int(np.round(3. * tau))
        except IndexError:
            i_start = 0
        # i_start = np.sum(mask_zero) - 2*N_traces
    else:
        i_start = i_start_list[ifol]

    N_valid_turns = np.sum(mask_zero)
    if i_start<0 or  i_start+N_traces > N_valid_turns:
        i_start = N_valid_turns-N_traces-1

    for i_trace in range(i_start, i_start+N_traces):
        wx_trace_filtered = savgol_filter(wx[:,i_trace], 31, 3)
        mask_filled = ob_slice.n_macroparticles_per_slice[:,i_trace]>0
        axtraces.plot(ob_slice.mean_z[mask_filled, i_trace],
                    wx_trace_filtered[mask_filled])

    axtraces.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    axtraces.grid(True, linestyle='--', alpha=0.5)
    axtraces.set_xlabel("z [m]")
    axtraces.set_ylabel("P.U. signal")
    axtraces.text(0.02, 0.02, 'Turns:\n%d - %d'%(i_start,
                i_start+N_traces-1),
            transform=axtraces.transAxes, ha='left', va='bottom',
            fontsize=10)

    titlestr = labels[ifol]
    if fname is not None:
        titlestr += (' - ' + fname)

    if flag_compact:
        plt.suptitle(titlestr,
            x=0.1,
            horizontalalignment='left')
    else:
        plt.suptitle(titlestr)

    # Get Qx Qs
    machine = LHC_custom.LHC(
              n_segments=1,
              machine_configuration=pars.machine_configuration,
              beta_x=pars.beta_x, beta_y=pars.beta_y,
              accQ_x=pars.Q_x, accQ_y=pars.Q_y,
              Qp_x=pars.Qp_x, Qp_y=pars.Qp_y,
              octupole_knob=pars.octupole_knob,
              optics_dict=None,
              V_RF=pars.V_RF
              )
    Qs = machine.longitudinal_map.Q_s
    Qx = machine.transverse_map.accQ_x
    frac_qx, _ = math.modf(Qx)

    axtext.text(0.5, 0.5,
            'Tune machine: %.4f'%frac_qx +\
            '\nSynchrotron tune: %.3fe-3 (V_RF: %.1f MV)'%(Qs*1e3, pars.V_RF*1e-6) +\
        '\nTune centroid: %.4f (%.2fe-3)\n'%(tune_centroid, 1e3*tune_centroid-frac_qx*1e3)+\
        f'Mode {i_mode}, {n_osc_axis[i_mode]:.2f} oscillations ' +\
        {False: "(most unstable)", True: "(forced)"}[forced] + '\n'+\
        'Tune mode (cos): %.4f (%.2fe-3)\n'%(tune_1mode_re, 1e3*tune_1mode_re-1e3*frac_qx) +\
        'Tune mode (sin): %.4f (%.2fe-3)\n'%(tune_1mode_im, 1e3*tune_1mode_im-1e3*frac_qx) +\
        f'Tau mode: {tau:.0f} turns',
        size=12, ha='center', va='center')
    axtext.axis('off')
    # These are the sin and cos components
    # (r+ji)(cos + j sin) + (r-ji)(cos - j sin)=
    # r cos + j r sin + ji cos - i sin | + r cos -j r sin -jicos -i sin = 
    # 2r cos - 2 i sin

    if fname is not None:
        figffts.savefig(fname+'_' + labels[ifol].replace(
            ' ', '_').replace('=', '').replace('-_', '')+'.png', dpi=200)

for ax in [ax11, ax12, ax13, axfft]:
    ax.grid(True, linestyle='--', alpha=0.5)

ax13.set_xlabel('Turn')
ax13.set_ylabel('Intrabunch\nactivity')
ax12.set_ylabel('Transverse\nemittance [um]')
ax11.set_ylabel('Transverse\nposition [mm]')
fig1.subplots_adjust(
        top=0.88,
        bottom=0.11,
        left=0.18,
        right=0.955,
        hspace=0.2,
        wspace=0.2)


leg = ax11.legend(prop={'size':10})
legfft = axfft.legend(prop={'size':10})
if fname is not None:
    fig1.savefig(fname+'.png', dpi=200)
    sio.savemat(fname+'_fit.mat', {
        'p_coeff': np.array(p_list)[:, 0]})
plt.show()
