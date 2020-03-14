import os
import numpy as np
import scipy.io as sio
import time

from scipy.constants import e as qe

from PyPARIS_sim_class import Simulation as sim_mod
import PyPARIS.util as pu


# start-settings-section
cos_amplitude = 1.00000000e-04
sin_amplitude = 0.00000000e+00
N_oscillations = 3.00000000e+00

flag_no_bunch_charge = False
flag_plots = True

sim_param_file = '../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = [
        '../Simulation_parameters_amend.py',
        'Simulation_parameters_amend_for_sin_response.py']
# end-settings-section

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

# Optionally remove charge from bunch
if flag_no_bunch_charge:
    bunch.particlenumber_per_mp = 1e-10

# Get slice centers
z_slices = slices_set.z_centers
N_slices = len(z_slices)

# Get z_step beween slices and define z_range
z_step = z_slices[1] - z_slices[0]
z_range = z_slices[-1] - z_slices[0] + z_step # Last term is to make the sampled 
                                              # sinusoids more orthogonal
# Generate ideal sinusoidal distortion 
x_ideal = (sin_amplitude * np.sin(2*np.pi*N_oscillations*z_slices/z_range)
         + cos_amplitude * np.cos(2*np.pi*N_oscillations*z_slices/z_range))

# Add sinusoidal distortion to particles
bunch.x += sin_amplitude * np.sin(2*np.pi*N_oscillations*bunch.z/z_range)
bunch.x += cos_amplitude * np.cos(2*np.pi*N_oscillations*bunch.z/z_range)

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
    if ss.slice_info == 'unsliced':
        continue
    temp_rho = 0.
    if np.mod(i_ss, 20)==0:
        print(("%d / %d"%(i_ss, N_slices)))
    for i_ee, ee in enumerate(sim_content.parent_eclouds):
        ee.track(ss)
        if i_ee == 0:
            temp_rho = ee.cloudsim.cloud_list[0].rho.copy()
        else:
            temp_rho += ee.cloudsim.cloud_list[0].rho.copy()
    dpx_slices.append(ss.mean_xp() * sim_content.n_segments)
    rho_slices.append(temp_rho)
dpx_slices = np.array(dpx_slices[::-1])
rho_slices = np.array(rho_slices[::-1])
t_end = time.mktime(time.localtime())
print(('Ecloud sim time %.2f s' % (t_end - t_start)))

dpx_slices_all_clouds = dpx_slices
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


sio.savemat('response.mat',{
    'z_slices': z_slices,
    'x_slices': x_slices,
    'x_ideal': x_ideal,
    'dpx_slices_all_clouds': dpx_slices_all_clouds,
    'xg': xg,
    'yg': yg,
    'rho_cut': rho_cut,
    })

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

    if i_yzero is not None:
        ax21.pcolormesh(z_slices, xg, rho_slices[:, :, i_yzero].T)
    ax21.plot(z_slices, x_slices, 'k', lw=2)
    ax22.plot(z_slices, dpx_slices)
    ax22.set_ylim(np.nanmax(np.abs(dpx_slices))*np.array([-1, 1]))
    ax22.grid(True)
    plt.show()



