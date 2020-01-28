import os
import numpy as np

from PyPARIS_sim_class import Simulation as sim_mod
import PyPARIS.util as pu


# Settings
cos_amplitude = 1.00000000e-04
sin_amplitude = 0.00000000e+00
N_oscillations = 0.00000000e+00

flag_no_bunch_charge = False
flag_plots = False

# Instantiate simulation
sim_content = sim_mod.Simulation(
    param_file='../reference_simulation/Simulation_parameters.py')

# Here sim_content.pp can be edited (directly and through files)
sim_content.pp.update(param_file='../Simulation_parameters_amend.py')
sim_content.pp.update(
        param_file='Simulation_parameters_amend_for_sin_response.py')

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
    if ss.macroparticlenumber:
        #if ss.mean_z() < 0:
        ss.x += sin_amplitude * np.sin(2*np.pi*N_oscillations*ss.z/z_range)
        ss.x += cos_amplitude * np.cos(2*np.pi*N_oscillations*ss.z/z_range)


