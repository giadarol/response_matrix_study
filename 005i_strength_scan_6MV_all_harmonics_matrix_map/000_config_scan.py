import os
import numpy as np

scan_folder_rel = 'simulations'

environment_preparation = f'''
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/venvs/py3/bin/activate
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/setpythopath
PYTHONPATH=$PYTHONPATH:{os.path.abspath('../')}
'''
# Last one is to get response matrix path

strength_scan = np.arange(0.1, 2.1, 0.1)

files_to_be_copied = [
        '../004_instability_simulation/000_simulation_matrix_map.py',
        '../004_instability_simulation/run_it_several_times',
        ]

settings_to_be_replaced_in = '000_simulation_matrix_map.py'


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
n_terms_to_be_kept = 200
n_tail_cut = 10
recenter_all_slices = True # Cancels initial kick from input

ecloud_strength_scale = {strength_scan[ii]:e}

sim_param_file = '../../../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = ['../../../Simulation_parameters_amend.py',
                    'Simulation_parameters_amend_for_matrixsim.py']

include_response_matrix = True
response_data_file = '../../../001_sin_response_scan/response_data.mat'

include_non_linear_map = True
field_map_file = '../../../003_generate_field_map/field_map.mat'
# end-settings-section'''

    sim_param_amend_curr= f'''
N_turns = 500

enable_transverse_damper = True
V_RF = 6e6'''

    with open(current_sim_abs_path
        + '/Simulation_parameters_amend_for_matrixsim.py', 'w') as fid:
        fid.write(sim_param_amend_curr)

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

bash run_it_several_times
'''
    with open(current_sim_abs_path + '/job.job', 'w') as fid:
       fid.write(job_content)

# Prepare htcondor cluster of jobs
import PyPARIS_sim_class.htcondor_config as htcc
htcc.htcondor_config(
        scan_folder_abs,
        time_requirement_days=2., # 120 minutes
        htcondor_files_in=scan_folder_abs)
