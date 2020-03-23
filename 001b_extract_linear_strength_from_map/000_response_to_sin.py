import os
import numpy as np
import scipy.io as sio
import time

from PyPARIS_sim_class import Simulation as sim_mod
import PyPARIS.util as pu

import PyECLOUD.myfilemanager as mfm

# start-settings-section
cos_amplitude = 1.00000000e-04
sin_amplitude = 0.00000000e+00
N_oscillations = 0.00000000e+00


flag_no_bunch_charge = False
flag_plots = True

sim_param_file = '../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = [
        '../Simulation_parameters_amend.py',
        'Simulation_parameters_amend_for_sin_response.py']

# field_map_file = '../003_generate_field_map/field_map.mat'
# fname = 'from_full_map'

field_map_file = '../003_generate_field_map/field_map_lin.mat'
fname = 'from_lin_map'
# end-settings-section

# Instantiate simulation
sim_content = sim_mod.Simulation(param_file=sim_param_file)

# Here sim_content.pp can be edited (directly and through files)
for ff in sim_param_amend_files:
    sim_content.pp.update(param_file=ff)

# Disable real e-clouds
sim_content.pp.enable_arc_dip = False
sim_content.pp.enable_arc_quad = False

# Add ring of CPU information (mimicking the master core)
pu.get_sim_instance(sim_content,
        N_cores_pretend=sim_content.pp.n_segments,
        id_pretend=sim_content.pp.n_segments-1,
        init_sim_objects_auto=False)
assert(sim_content.ring_of_CPUs.I_am_the_master)

# Initialize machine elements
sim_content.init_all()

# Initialize master to get the beam
if os.path.exists('simulation_status.sta'):
    os.remove('simulation_status.sta')
slices = sim_content.init_master()
N_slices = len(slices)

# Re-center all slices
for ss in slices:
    if ss.macroparticlenumber:
        ss.x -= ss.mean_x()
        ss.xp -= ss.mean_xp()

# Get slice centers
z_slices = np.array([ss.slice_info['z_bin_center'] for ss in slices])

# Get z_step beween slices and define z_range
z_step = z_slices[1] - z_slices[0]
z_range = z_slices[-1] - z_slices[0] + z_step # Last term is to make the sampled 
                                              # sinusoids more orthogonal
# Generate ideal sinusoidal distortion 
x_ideal = (sin_amplitude * np.sin(2*np.pi*N_oscillations*z_slices/z_range)
         + cos_amplitude * np.cos(2*np.pi*N_oscillations*z_slices/z_range))

# Add sinusoidal distortion to particles
for ss in slices:
    if ss.macroparticlenumber>0:
        #if ss.mean_z() < 0:
        ss.x += sin_amplitude * np.sin(2*np.pi*N_oscillations*ss.z/z_range)
        ss.x += cos_amplitude * np.cos(2*np.pi*N_oscillations*ss.z/z_range)

# Measure
x_slices = np.array([ss.mean_x() for ss in slices])
int_slices = np.array([ss.intensity for ss in slices])


obfmap = mfm.myloadmat_to_obj(field_map_file)
from PyECLOUD.Transverse_Efield_map_for_frozen_cloud import Transverse_Efield_map
fmap = Transverse_Efield_map(
    xg=obfmap.xg,
    yg=obfmap.yg,
    Ex=obfmap.Ex_L_map,
    Ey=obfmap.Ey_L_map,
    L_interaction=1./sim_content.n_segments,
    slicer=None,
    flag_clean_slices=False,
    wrt_slice_centroid=False,
    x_beam_offset=0.,
    y_beam_offset=0.,
    slice_by_slice_mode=True)

assert(len(sim_content.parent_eclouds)==0)
sim_content.parent_eclouds.append(fmap)

# Simulate e-cloud interactions
t_start = time.mktime(time.localtime())
dpx_slices = []
for i_ss, ss in enumerate(slices[::-1]):
    if np.mod(i_ss, 20)==0:
        print(("%d / %d"%(i_ss, N_slices)))
    for i_ee, ee in enumerate(sim_content.parent_eclouds):
        ee.track(ss)
    dpx_slices.append(ss.mean_xp())
dpx_slices = np.array(dpx_slices[::-1])
t_end = time.mktime(time.localtime())
print(('Ecloud sim time %.2f s' % (t_end - t_start)))

dpx_slices_all_clouds = dpx_slices * sim_content.n_segments

# Savings and plots
xg = obfmap.xg
yg = obfmap.yg

rho_cut = np.zeros((len(z_slices), len(xg)))

assert(N_oscillations == 0)
k_z = dpx_slices_all_clouds / x_slices

sio.savemat(fname+'_response.mat',{
    'z_slices': z_slices,
    'x_slices': x_slices,
    'x_ideal': x_ideal,
    'dpx_slices_all_clouds': dpx_slices_all_clouds,
    'xg': xg,
    'yg': yg,
    'rho_cut': rho_cut,
    })

sio.savemat(fname + '_linear_strength.mat', {
    'z_slices': z_slices,
    'k_z_integrated': k_z})

if flag_plots:
    import matplotlib.pyplot as plt
    plt.close('all')
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(3,1,1)
    ax2 = fig1.add_subplot(3,1,2, sharex=ax1)
    ax3 = fig1.add_subplot(3,1,3, sharex=ax1)

    ax1.plot(z_slices, int_slices)
    ax2.plot(z_slices, x_slices)
    ax3.plot(z_slices, dpx_slices)

    for ax in [ax1, ax2, ax3]:
        ax.grid(True)


    fig2 = plt.figure(2)
    ax21 = fig2.add_subplot(2,1,1)
    ax22 = fig2.add_subplot(2,1,2, sharex=ax21)

    ax21.pcolormesh(z_slices, xg, rho_cut.T)
    ax21.plot(z_slices, x_slices, 'k', lw=2)
    ax22.plot(z_slices, dpx_slices)
    ax22.set_ylim(np.nanmax(np.abs(dpx_slices))*np.array([-1, 1]))
    ax22.grid(True)
    plt.show()



