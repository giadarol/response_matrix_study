import os
import time
import glob
import numpy as np
import scipy.io as sio

from scipy.constants import e as qe

from PyPARIS_sim_class import Simulation as sim_mod
import PyPARIS.util as pu

import PyPARIS_sim_class.propsort as ps
import PyECLOUD.myfilemanager as mfm

flag_trace_from_simulation = True
folder_curr_sim = '../reference_simulation'
i_sim_trace = 1200 
fname_out = f'refsim_turn{i_sim_trace}.mat'


sim_param_file = '../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = ['../Simulation_parameters_amend.py']

# Instantiate simulation
sim_content = sim_mod.Simulation(param_file=sim_param_file)

# Here sim_content.pp can be edited (directly and through files)
for ff in sim_param_amend_files:
    sim_content.pp.update(param_file=ff)

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
sim_content.init_master()

# Get bunch and slicer
bunch = sim_content.bunch
slicer = sim_content.slicer

# Recenter all slices
slices_set = bunch.get_slices(slicer, statistics=True)
for ii in range(slices_set.n_slices):
    ix = slices_set.particle_indices_of_slice(ii)
    if len(ix) > 0:
        bunch.x[ix] -= np.mean(bunch.x[ix])
        bunch.xp[ix] -= np.mean(bunch.xp[ix])

# Apply distorsion from simulation
if flag_trace_from_simulation:
    sim_curr_list_slice_ev = ps.sort_properly(glob.glob(folder_curr_sim+'/slice_evolution_*.h5'))
    ob_slice = mfm.monitorh5list_to_obj(sim_curr_list_slice_ev, key='Slices', flag_transpose=True)

    x_trace = ob_slice.mean_x[:, i_sim_trace]
    z_trace = ob_slice.mean_z[:, i_sim_trace]
    int_trace = ob_slice.n_macroparticles_per_slice[:, i_sim_trace]
    mask_keep = int_trace > 0.0
    x_trace_masked = x_trace[mask_keep]
    z_trace_masked = z_trace[mask_keep]
    int_trace_masked = int_trace[mask_keep]
    assert(np.min(np.diff(z_trace_masked)) > 0)

    bunch.x += np.interp(bunch.z, z_trace_masked, x_trace_masked)

# Get slice centers
z_slices = slices_set.z_centers
N_slices = len(z_slices)

# Measure
bunch.clean_slices()
slices_set = bunch.get_slices(slicer, statistics=True)
x_slices = slices_set.mean_x
int_slices = slices_set.lambda_bins()/qe

# Apply impedance
for imp in sim_content.impedances:
    imp.track(bunch)

# Extract slice objects
slices = bunch.extract_slices(slicer)

# Simulate e-cloud interactions
t_start = time.mktime(time.localtime())
dpx_slices = []
rho_slices = []
for i_ss, ss in enumerate(slices[::-1]):
    temp_rho = 0.
    if np.mod(i_ss, 20)==0:
        print(("%d / %d"%(i_ss, N_slices)))
    for i_ee, ee in enumerate(sim_content.parent_eclouds):
        ee.track(ss)
        if i_ee == 0:
            temp_rho = ee.cloudsim.cloud_list[0].rho.copy()
        else:
            temp_rho += ee.cloudsim.cloud_list[0].rho.copy()
    dpx_slices.append(ss.mean_xp())
    rho_slices.append(temp_rho)
dpx_slices = np.array(dpx_slices[::-1])
rho_slices = np.array(rho_slices[::-1])
t_end = time.mktime(time.localtime())
print(('Ecloud sim time %.2f s' % (t_end - t_start)))

# Savings and plots
if len(sim_content.parent_eclouds) > 0:
    first_ecloud = sim_content.parent_eclouds[0]
    xg = first_ecloud.cloudsim.spacech_ele.xg
    yg = first_ecloud.cloudsim.spacech_ele.yg

    i_yzero = np.argmin(np.abs(yg))
    rho_cut = rho_slices[:, :, i_yzero]
else:
    xg = 0.
    yg = 0.
    rho_cut = 0.
    i_yzero = None

sio.savemat(fname_out,{
    'z_slices': z_slices,
    'x_slices': x_slices,
    'dpx_slices': dpx_slices,
    'int_slices': int_slices,
    'xg': xg,
    'yg': yg,
    'rho_cut': rho_cut,
    })

import matplotlib.pyplot as plt
plt.close('all')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(z_trace_masked, x_trace_masked)
ax2.plot(z_slices, x_slices)

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.plot(z_slices, dpx_slices)


fig20 = plt.figure(20)
ax21 = fig20.add_subplot(2,1,1)
ax22 = fig20.add_subplot(2,1,2, sharex=ax21)

if i_yzero is not None:
    ax21.pcolormesh(z_slices, xg, rho_slices[:, :, i_yzero].T)
ax21.plot(z_slices, x_slices, 'k', lw=2)
ax22.plot(z_slices, dpx_slices)
ax22.set_ylim(np.nanmax(np.abs(dpx_slices))*np.array([-1, 1]))
ax22.grid(True)

plt.show()
