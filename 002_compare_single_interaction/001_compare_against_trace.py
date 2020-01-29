import os

import numpy as np

from PyPARIS_sim_class import Simulation as sim_mod
import PyPARIS.util as pu
import PyECLOUD.myfilemanager as mfm

# Import response_matrix
import sys
sys.path.append('../')
import response_matrix.response_matrix as rm

test_data_file = './refsim_turn302.mat'
n_terms_to_be_kept = 12
n_tail_cut = 10
response_data_file = '../001_sin_response_scan/response_data.mat'

sim_param_file = '../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = ['../Simulation_parameters_amend.py']

# Instantiate simulation
sim_content = sim_mod.Simulation(param_file=sim_param_file)

# Here sim_content.pp can be edited (directly and through files)
for ff in sim_param_amend_files:
    sim_content.pp.update(param_file=ff)

# Disable real e-clouds
sim_content.pp.enable_arc_dip = False
sim_content.pp.enable_arc_quad = False

# Add ring of CPU information
ring_cpu = pu.get_serial_CPUring(sim_content,
        init_sim_objects_auto=False)
assert(sim_content.ring_of_CPUs.I_am_the_master)

# Initialize machine elements
sim_content.init_all()

# Initialize master to get the beam
if os.path.exists('simulation_status.sta'):
    os.remove('simulation_status.sta')

# Initialize beam
sim_content.init_master()

# Get bunch and slicer
bunch = sim_content.bunch
slicer = sim_content.slicer

# Recenter all slices
slices = bunch.get_slices(slicer)
for ii in range(slices.n_slices):
    ix = slices.particle_indices_of_slice(ii)
    if len(ix) > 0:
        bunch.x[ix] -= np.mean(bunch.x[ix])
        bunch.xp[ix] -= np.mean(bunch.xp[ix])

# Build matrix
respmat = rm.ResponseMatrix(
        slicer=slicer,
        response_data_file=response_data_file,
        coord='x',
        kick_factor=1./sim_content.n_segments,
        n_terms_to_be_kept=n_terms_to_be_kept,
        n_tail_cut=n_tail_cut)

# Get simulation data
obsim = mfm.myloadmat_to_obj(test_data_file)
x_test = obsim.x_slices
int_test = obsim.int_slices
x_test[np.isnan(x_test)] = 0.

# Distort bunch
bunch.x = bunch.x + np.interp(bunch.z, respmat.z_resp, x_test)

# Apply matrix
respmat.track(bunch)

# Measure kicks
bunch.clean_slices()
slices_test = bunch.get_slices(slicer, statistics=['mean_x', 'mean_xp'])

# Plots
import matplotlib.pyplot as plt
plt.close('all')
z_resp = respmat.z_resp

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(z_resp, x_test)
ax2.plot(slices_test.z_centers, slices_test.mean_x, '.')

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.plot(z_resp, obsim.dpx_slices, label='sim')
ax3.plot(z_resp, respmat.response_to_slice_array(x_test), label='lin2')
ax3.plot(slices_test.z_centers, slices_test.mean_xp, label='from matrix')
ax3.legend()

xg = obsim.xg
yg = obsim.yg
i_yzero = np.argmin(np.abs(xg))

fig20 = plt.figure(20)
ax21 = fig20.add_subplot(2,1,1)
ax22 = fig20.add_subplot(2,1,2, sharex=ax21)

ax21.pcolormesh(obsim.z_slices, xg, obsim.rho_cut.T)
ax21.plot(obsim.z_slices, obsim.x_slices, 'k', lw=2)
ax22.plot(obsim.z_slices, obsim.dpx_slices)
ax22.set_ylim(np.nanmax(np.abs(obsim.dpx_slices))*np.array([-1, 1]))
ax22.grid(True)


plt.show()
