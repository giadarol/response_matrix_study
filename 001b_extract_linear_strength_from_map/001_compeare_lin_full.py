import matplotlib.pyplot as plt

import PyECLOUD.myfilemanager as mfm

obfull = mfm.myloadmat_to_obj('./from_full_map_linear_strength.mat')
oblin = mfm.myloadmat_to_obj('./from_lin_map_linear_strength.mat')

plt.close('all')
fig1 = plt.figure(1)
plt.plot(obfull.z_slices, obfull.k_z_integrated)
plt.plot(obfull.z_slices, oblin.k_z_integrated)
plt.show()
