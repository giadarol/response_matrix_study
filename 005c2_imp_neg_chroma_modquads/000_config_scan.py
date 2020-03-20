import os
import numpy as np

scan_folder_rel = 'simulations_corrected'

environment_preparation = f'''
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/venvs/py3/bin/activate
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/setpythopath
PYTHONPATH=$PYTHONPATH:{os.path.abspath('../')}
'''
# Last one is to get response matrix path

strength_scan = np.arange(0, 1., 0.005)

files_to_be_copied = [
        '../005_impedance_instability_simulation/000_simulation_imped.py',
        '../005_impedance_instability_simulation/Simulation_parameters_amend_for_impsim.py',
        ]

settings_to_be_replaced_in = '000_simulation_imped.py'


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
recenter_all_slices = True # Cancels initial kick from input

strength_scale = {strength_scan[ii]:.2e}

Qp_x = 0.
alpha_N = []
beta_N = [0, -5.*(-4*np.pi/97.2)]

n_segments = 15

flag_enable_multiple_runs = False

sim_param_file = '../../../reference_simulation/Simulation_parameters.py'
sim_param_amend_files = ['../../../Simulation_parameters_amend.py',
                    'Simulation_parameters_amend_for_impsim.py']
# end-settings-section'''

    # sim_param_amend_curr= f''' '''

    # with open(current_sim_abs_path
    #     + '/Simulation_parameters_amend_for_matrixsim.py', 'w') as fid:
    #     fid.write(sim_param_amend_curr)

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

python 000_simulation_imped.py
'''
    with open(current_sim_abs_path + '/job.job', 'w') as fid:
       fid.write(job_content)

# Prepare htcondor cluster of jobs
import PyPARIS_sim_class.htcondor_config as htcc
htcc.htcondor_config(
        scan_folder_abs,
        time_requirement_days=1.,
        htcondor_files_in=scan_folder_abs)
