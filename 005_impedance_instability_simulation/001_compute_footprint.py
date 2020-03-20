import os
import time

import numpy as np

import PyECLOUD.myfilemanager as mfm
from PyPARIS_sim_class import Simulation as sim_mod
import PyPARIS.util as pu
import PyPARIS_sim_class.frequency_analysis as fa


# Import modulated_quadrupol
import sys
sys.path.append('../')
import response_matrix.modulated_quadrupole as mq

# start-settings-section
recenter_all_slices = False # Cancels initial kick from input

strength_scale = 1.

Qp_x = 0.
alpha_N = [0, 0, 0.]
beta_N = [0., -5.*(-4*np.pi/92.7)]

n_segments = 8

flag_enable_multiple_runs = False

sim_param_file = '../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = ['../Simulation_parameters_amend.py',
                    'Simulation_parameters_amend_for_impsim.py']

N_turns_footprint = 1024
N_particles_footprint = 5000

include_non_linear_map = False
# end-settings-section

# Instantiate simulation
sim_content = sim_mod.Simulation(param_file=sim_param_file)

# Here sim_content.pp can be edited (directly and through files)
for ff in sim_param_amend_files:
    sim_content.pp.update(param_file=ff)

# Set chromaticity
sim_content.pp.Qp_x = Qp_x
sim_content.pp.n_segments = n_segments

# Switch off damper and impedance for footprint
sim_content.pp.enable_transverse_damper = False
sim_content.pp.enable_impedance = False

# Add ring of CPU information
ring_cpu = pu.get_serial_CPUring(sim_content,
        init_sim_objects_auto=False)
assert(sim_content.ring_of_CPUs.I_am_the_master)

# Initialize machine elements
sim_content.init_all()

# Remove longitudinal map for footprint
sim_content.machine.one_turn_map.remove(
        sim_content.machine.longitudinal_map)

# Initialize beam, slicer, monitors, multijob mode
if os.path.exists('simulation_status.sta'):
    os.remove('simulation_status.sta')
sim_content.init_master(generate_bunch=False,
        prepare_monitors=False)

# Get bunch, slicer, machine and monitors
slicer = sim_content.slicer
machine = sim_content.machine

# Add modulated quadrupole
if len(alpha_N)>0 or len(beta_N)>0:
    mquad = mq.ModulatedQuadrupole(coord='x',
            alpha_N=np.array(alpha_N)/sim_content.pp.n_segments,
            beta_N=np.array(beta_N)/sim_content.pp.n_segments)
    machine.install_after_each_transverse_segment(mquad)

# Generate bunch for footprint
pp = sim_content.pp
bunch = sim_content.machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles=N_particles_footprint,
    intensity=pp.intensity,
    epsn_x=pp.epsn_x,
    epsn_y=pp.epsn_y,
    sigma_z=pp.sigma_z,
)

# Slice the bunch
slices = bunch.extract_slices(slicer)

# Recenter all slices
if recenter_all_slices and sim_content.SimSt.first_run:
    slices = bunch.get_slices(slicer)
    for ii in range(slices.n_slices):
        ix = slices.particle_indices_of_slice(ii)
        if len(ix) > 0:
            bunch.x[ix] -= np.mean(bunch.x[ix])
            bunch.xp[ix] -= np.mean(bunch.xp[ix])
            bunch.y[ix] -= np.mean(bunch.y[ix])
            bunch.yp[ix] -= np.mean(bunch.yp[ix])

# Introduce non-linear field map
if include_non_linear_map:

    obfmap = mfm.myloadmat_to_obj(field_map_file)
    from PyECLOUD.Transverse_Efield_map_for_frozen_cloud import Transverse_Efield_map
    fmap = Transverse_Efield_map(
        xg=obfmap.xg,
        yg=obfmap.yg,
        Ex=obfmap.Ex_L_map,
        Ey=obfmap.Ey_L_map,
        L_interaction=1./sim_content.n_segments*ecloud_strength_scale,
        slicer=slicer,
        flag_clean_slices=False,
        wrt_slice_centroid=False, # Only for footprint
        x_beam_offset=0.,
        y_beam_offset=0.,
        slice_by_slice_mode=False)
    machine.install_after_each_transverse_segment(fmap)

# Prepare to save turn-by-turn data
recorded_particles = sim_mod.ParticleTrajectories(
                N_particles_footprint, N_turns_footprint)

# Simulate
slice_x_list = []
for i_turn in range(N_turns_footprint):
    print('%s Turn %d/%d' % (
        time.strftime("%d/%m/%Y %H:%M:%S",time.localtime()),
        i_turn, N_turns_footprint
        ))
    recorded_particles.dump(bunch)
    machine.track(bunch)

# Compute tunes
fa.get_tunes(recorded_particles,
        filename_output='footprint.h5')

