import numpy as np

from scipy.constants import c as clight
import matplotlib.pyplot as plt

from Impedance import imp_model_resonator, freq_param

beta = 1.
# Broad-band impedance
Rt = 25e6 # shunt impedance in MOhm/m
fr = 2e9 # cutoff frequency in Hz
Q = 1 # quality factor

Nb_vect = [6e11] # np.arange(0, 10e11, 0.1e11)

imp_mod, _ = imp_model_resonator(
        Rlist=Rt, frlist=fr, Qlist=Q,beta=beta,
        fpar=freq_param(fmin=10,fmax=1e13,ftypescan=2,
                nflog=10,fminrefine=fr*0.1,fmaxrefine=fr*10,
                nrefine=200,fadded=[]),
                listcomp=['Zxdip'])

from DELPHI import compute_impedance_matrix, computes_coef
from DELPHI import eigenmodesDELPHI
from DELPHI import longdistribution_decomp

lmax = 2
nmax = 0
nx = 0 # Coupled-bunch mode
M = 1 # Number of bunches
omegaksi = 0. # Chromatic shift
omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
Q_frac = .31
Q_full = 62.31
tau_b = 1e-9
a_coeff = 8.
b_coeff = a_coeff
g_0 = a_coeff/np.pi/tau_b**2
gamma = 6927.62871617
omega_s = 0.001909*omega0

g,a,b = longdistribution_decomp(tau_b, typelong='Gaussian')


MM = compute_impedance_matrix(
        lmax, nmax, nx, M, omegaksi, omega0, Q_frac,
        a_coeff, b_coeff, tau_b, np.array([g_0]),
        Z=imp_mod[0].func, freqZ=imp_mod[0].var,
        flag_trapz=1, abseps=1,
        lmaxold=-1, nmaxold=-1, couplold=None)

eigenval_list = []

for Nb in Nb_vect:
    kdammp, kimp = computes_coef(
            f0=omega0/2/np.pi, dmax=0,
            b=a_coeff,g0=g_0,
            dnormfactor=1,
            taub=tau_b,
            dphase=1,M=M,
            Nb=Nb,
            gamma=gamma,
            Q=Q_full,
            particle='proton')
    print('k_imp=',kimp)
    eigenval, _ = eigenmodesDELPHI(
        lmax,nmax,matdamper=0*MM,matZ=MM,
        coefdamper=0,coefZ=kimp,
        omegas=omega_s,flageigenvect=False)

    eigenval_list.append(eigenval)
plt.close('all')
plt.figure(1)
plt.semilogx(imp_mod[0].var, imp_mod[0].func[:,0], '.-', label='Re')
plt.semilogx(imp_mod[0].var, imp_mod[0].func[:,1], '.-', label='Im')
plt.legend()

eigval_mat = np.array(eigenval_list)

plt.figure(2)
plt.plot(Nb_vect, np.real(eigval_mat)/omega_s, '.b')

plt.figure(3)
plt.plot(Nb_vect, np.imag(eigval_mat), '.b')
plt.show()
