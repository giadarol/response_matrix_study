import os
import numpy as np

scan_folder_rel = 'simulations'

environment_preparation = f'''
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/venvs/py3/bin/activate
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/setpythopath
PYTHONPATH=$PYTHONPATH:{os.path.abspath('../')}
'''
# Last one is to get response matrix path

strength_scan = np.arange(0.1, 1.1, 0.1)

files_to_be_copied = [
        '../004_instability_simulation/001_compute_footprint.py']

settings_to_be_replaced_in = '001_compute_footprint.py'


scan_folder_abs = os.getcwd() + '/' + scan_folder_rel
os.mkdir(scan_folder_abs)

# prepare scan
for ii in range(len(strength_scan)):

    # Make directory
    current_sim_ident= f'strength_{strength_scan[ii]:.2e}'
    print(current_sim_ident)
    current_sim_abs_path = scan_folder_abs+'/'+current_sim_ident
    os.mkdir(current_sim_abs_path)

    # Copy files
    for ff in files_to_be_copied:
        os.system(f'cp {ff} {current_sim_abs_path}')

    # Replace settings section
    settings_section = f'''# start-settings-section
recenter_all_slices = False # Cancels initial kick from input

ecloud_strength_scale = {strength_scan[ii]:e}

sim_param_file = '../../../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = ['../../../Simulation_parameters_amend.py']

include_non_linear_map = True
field_map_file = '../../../003_generate_field_map/field_map.mat'

N_turns_footprint = 1024
N_particles_footprint = 5000
# end-settings-section'''


    with open(current_sim_abs_path+'/'+settings_to_be_replaced_in, 'r') as fid:
        lines = fid.readlines()
    istart = np.where([ss.startswith('# start-settings-section') for ss in lines])[0][0]
    iend = np.where([ss.startswith('# end-settings-section') for ss in lines])[0][0]
    with open(current_sim_abs_path+'/'+settings_to_be_replaced_in, 'w') as fid:
        fid.writelines(lines[:istart])
        fid.write(settings_section + '\n')
        fid.writelines(lines[iend+1:])

    # Prepare job script
    job_content = f'''#!/bin/bash

{environment_preparation}

# Environment preparation
echo PYTHONPATH=$PYTHONPATH
echo which python
which python

cd {current_sim_abs_path}

python 001_compute_footprint.py
'''
    with open(current_sim_abs_path + '/job.job', 'w') as fid:
       fid.write(job_content)

# Prepare htcondor cluster of jobs
import PyPARIS_sim_class.htcondor_config as htcc
htcc.htcondor_config(
        scan_folder_abs,
        time_requirement_days=0.15,
        htcondor_files_in=scan_folder_abs)
