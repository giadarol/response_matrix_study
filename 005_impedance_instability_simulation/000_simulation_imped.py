import os
import time

import numpy as np

import PyECLOUD.myfilemanager as mfm
from PyPARIS_sim_class import Simulation as sim_mod
import PyPARIS.util as pu


# start-settings-section
recenter_all_slices = True # Cancels initial kick from input

strength_scale = 1.

sim_param_file = '../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = ['../Simulation_parameters_amend.py',
                    'Simulation_parameters_amend_for_impsim.py']
# end-settings-section


# Instantiate simulation
sim_content = sim_mod.Simulation(param_file=sim_param_file)

# Here sim_content.pp can be edited (directly and through files)
for ff in sim_param_amend_files:
    sim_content.pp.update(param_file=ff)


# Add ring of CPU information
ring_cpu = pu.get_serial_CPUring(sim_content,
        init_sim_objects_auto=False)
assert(sim_content.ring_of_CPUs.I_am_the_master)


# Adjust impedance strength
sim_content.pp.resonator_R_shunt *= strength_scale

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

# Simulate
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