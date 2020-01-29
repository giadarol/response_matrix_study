import os

import numpy as np

from PyPARIS_sim_class import Simulation as sim_mod
import PyPARIS.util as pu
import PyECLOUD.myfilemanager as mfm

# Import response_matrix
import sys
sys.path.append('../')
import response_matrix.response_matrix as rm

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
N_slices = len(slices)

# Re-center all slices
for ss in slices:
    if ss.macroparticlenumber:
        ss.x -= ss.mean_x()
        ss.xp -= ss.mean_xp()

# Build matrix
respmat = rm.ResponseMatrix(
        slicer=sim_content.slicer,
        response_data_file=response_data_file,
        coord='x',
        kick_factor=1./sim_content.n_segments,
        n_terms_to_be_kept=n_terms_to_be_kept,
        n_tail_cut=n_tail_cut)
