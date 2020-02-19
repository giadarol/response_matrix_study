import numpy as np

from scipy.constants import c as clight
import matplotlib.pyplot as plt

from Impedance import imp_model_resonator, freq_param

beta = 1.
# Broad-band impedance
Rt = 25e6 # shunt impedance in MOhm/m
fr = 2e9 # cutoff frequency in Hz
Q = 1 # quality factor
imp_mod, _ = imp_model_resonator(
        Rlist=Rt, frlist=fr, Qlist=Q,beta=beta,
        fpar=freq_param(fmin=10,fmax=1e13,ftypescan=2,
                nflog=10,fminrefine=fr*0.1,fmaxrefine=fr*10,
                nrefine=200,fadded=[]),
                listcomp=['Zxdip'])

from DELPHI import compute_impedance_matrix, computes_coef
nx = 0 # Coupled-bunch mode
M = 1 # Number of bunches
omegaksi = 0. # Chromatic shift
omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
Q_frac = .31
tau_b = 1e-9/4

b_coef = a_coeff

plt.close('all')
plt.figure()
plt.semilogx(imp_mod[0].var, imp_mod[0].func[:,0], '.-', label='Re')
plt.semilogx(imp_mod[0].var, imp_mod[0].func[:,1], '.-', label='Im')
plt.legend()
plt.show()
