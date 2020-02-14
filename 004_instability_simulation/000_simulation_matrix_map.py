import os
import time

import numpy as np

import PyECLOUD.myfilemanager as mfm
from PyPARIS_sim_class import Simulation as sim_mod
import PyPARIS.util as pu

# Import response_matrix
import sys
sys.path.append('../')
import response_matrix.response_matrix as rm


# start-settings-section
n_terms_to_be_kept = 200
n_tail_cut = 10
recenter_all_slices = False # Cancels initial kick from input

ecloud_strength_scale = 1.

sim_param_file = '../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = ['../Simulation_parameters_amend.py',
                    'Simulation_parameters_amend_for_matrixsim.py']

include_response_matrix = True
response_data_file = '../001_sin_response_scan/response_data.mat'

include_non_linear_map = False
field_map_file = '../003_generate_field_map/field_map.mat'
# end-settings-section


# Instantiate simulation
sim_content = sim_mod.Simulation(param_file=sim_param_file)

# Here sim_content.pp can be edited (directly and through files)
for ff in sim_param_amend_files:
    sim_content.pp.update(param_file=ff)

# Disable real e-clouds and impedance
sim_content.pp.enable_arc_dip = False
sim_content.pp.enable_arc_quad = False
sim_content.pp.enable_impedance = False

# Add ring of CPU information
ring_cpu = pu.get_serial_CPUring(sim_content,
        init_sim_objects_auto=False)
assert(sim_content.ring_of_CPUs.I_am_the_master)

# Initialize machine elements
sim_content.init_all()

# Initialize beam, slicer, monitors, multijob mode
sim_content.init_master()

# Get bunch, slicer, machine and monitors
bunch = sim_content.bunch
slicer = sim_content.slicer
machine = sim_content.machine
bunch_monitor = sim_content.bunch_monitor
slice_monitor = sim_content.slice_monitor

# Recenter all slices
if recenter_all_slices and sim_content.SimSt.first_run:
    slices = bunch.get_slices(slicer)
    for ii in range(slices.n_slices):
        ix = slices.particle_indices_of_slice(ii)
        if len(ix) > 0:
            bunch.x[ix] -= np.mean(bunch.x[ix])
            bunch.xp[ix] -= np.mean(bunch.xp[ix])

# Build matrix
if include_response_matrix:
    respmat = rm.ResponseMatrix(
        slicer=slicer,
        response_data_file=response_data_file,
        coord='x',
        kick_factor=1./sim_content.n_segments*ecloud_strength_scale,
        n_terms_to_be_kept=n_terms_to_be_kept,
        n_tail_cut=n_tail_cut)
    machine.install_after_each_transverse_segment(respmat)

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
        wrt_slice_centroid=True,
        x_beam_offset=0.,
        y_beam_offset=0.,
        slice_by_slice_mode=False)
    machine.install_after_each_transverse_segment(fmap)


# Simulate
slice_x_list = []
for i_turn in range(sim_content.N_turns):
    print('%s Turn %d' % (time.strftime("%d/%m/%Y %H:%M:%S", time.localtime()), i_turn))

    bunch_monitor.dump(bunch)

    slices = bunch.get_slices(slicer, statistics=['mean_x'])
    slice_monitor.dump(bunch)

    machine.track(bunch)

    if sim_content._check_stop_conditions():
        os.system('touch met_stop_condition')
        break

sim_content._finalize_multijob_mode()
