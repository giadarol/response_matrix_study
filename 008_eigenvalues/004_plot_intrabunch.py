import pickle

import numpy as np
import scipy
from scipy.constants import c as clight
import scipy.io as sio

import PyECLOUD.mystyle as ms

pkl_fname = 'mode_coupling_matrix.pkl'

l_min = -7
l_max = 7
m_max = 20
N_max = 49
abs_min_imag_unstab = 1.


with open(pkl_fname, 'rb') as fid:
    MM_orig = pickle.load(fid)

MM_obj = MM_orig.get_sub_matrix(l_min, l_max, m_max, N_max)

omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
omega_s = 4.9e-3*omega0

# Mode coupling test
strength = 1.28
Omega_mat, evect = MM_obj.compute_mode_complex_freq(omega_s,
        rescale_by=[strength], flag_eigenvectors=True)
i_most_unstable = np.argmin(np.imag(Omega_mat))

def R_computation_v2(eigenvector, ldown, lup, nmax, r, a, b, taub, beta=1):
    ''' Computes R(tau) as defined in as in N. Mounet, "DELPHI_expanded", slide 22

    Inputs
    ======
    - eigenvector: numpy array
        The mode eigenvector computed with DELPHI
    - ldown: int
        The lowest azimuthal mode number computed
    - lup: int
        The highest azimuthal mode number computed
    - nmax: int
        The number of radial modes computed
    - r: numpy array
        The time sampling of the signal
    - a, b: floats
        The coefficient given by DELPHI.py function longdistribution_decomp
        For a Gaussian distribution, a = b = 8
    - taub: float
        Bunch length in seconds
    - beta: float (between 0 and 1)
        Particle speed w.r.t to light speed

    Outputs
    =======
    - Rl_table: numpy array

    Example
    =======
    tau = np.linspace(0, 4.0e-9, 1000)
    r = 299792458 * beta * tau

    # For standard DELPHI simulations, we have lup = -ldown = nmax
    # The values can therefore be found from the size of the eigenvector array
    # For example we can have (ldown, lup, nmax) = (-1, 1, 1) ... (-15, 15, 15)

    lmax = ((-3 + np.sqrt(9 - 8*(1.0 - len(eigenvector))))/4).astype(int)
    lup = lmax
    ldown = -lmax
    nmax = lmax

    Rl_table = R_computation_v2(eigenvector, ldown, lup, nmax, r, a, b, SPS.taub, SPS.beta)

    '''


    clight = 299792458
    #Rl is an interdemediate variable for the calculations, see N. Mounet, slide 39
    #Rl_table contains the Rl for all the l from lmax to -lmax-1
    Rl = np.zeros(len(r), dtype=complex)
    Rl_table = np.zeros(((lup-ldown+1), len(r)), dtype=complex)

    #Rsum contains the result of the addition of all Rl
    Rsum = np.zeros(len(r), dtype=complex)

    #First the computation of each Rl
    for l in range(lup,ldown-1,-1):

        for n in range(nmax+1):

            Rl += (eigenvector[n + (l-ldown)*(nmax+1)]
                   * scipy.special.eval_genlaguerre(n, abs(l), a * ((r / (beta*clight*taub))) ** 2))

        Rl *= (r / (beta*clight*taub)) ** abs(l) * np.exp(-b * ((r / (beta*clight*taub))) ** 2)

        #Each Rl is stored then erased
        Rl_table[l-ldown, :] = Rl

        Rl = np.zeros(len(r),dtype=complex)

    return Rl_table


def headtail_signal(Rl_table, max_freq, nb_points, lup, ldown, eigenvalue,
                    r, tune, Qp, eta, radius, beta=1, n_signals=10):
    '''
    Inputs
    ======
    - Rl_table: numpy array
        Table with Rl values, computed with the R_computation_v2 function
    - max_freq: float
        Maximum frequency of the sampling (in Hz)
    - nb_points: int
        Number of points in the frequency sampling
    - lup, ldown: int
        The lowest and highest azimuthal mode numbers computed by DELPHI
    - eigenvalue: complex
        The eigenvalue of the mode investigated
    - r: numpy array
        The time sampling of the signal
    - tune: float
        The beam tune
    - Qp: float
        The beam chromaticity (in unnormalized units)
    - eta: float
        Beam slippage factor
    - radius: float
        machine radius
    - beta: float (between 0 and 1)
        Particle speed w.r.t to light speed
    -n_signals: int
        Number of headtail traces to compute

    Outputs
    =======
    - freq = omega/(2*np.pi): numpy array
        Signal spectrum frequency array
    - lamba1tot: numpy array
        Complex signal spectrum
    - time: numpy array
        Time domain signal times
    - list_signals: numpy array
        List of time domain signals


    Example
    =======
    # the sampling
    tau = np.linspace(0, 4.0e-9, 500)
    r = 299792458 * beta * tau

    # For standard DELPHI simulations, we have lup = -ldown = nmax
    # The values can therefore be found from the size of the eigenvector array
    # For example we can have (ldown, lup, nmax) = (-1, 1, 1) ... (-15, 15, 15)
    lmax = ((-3 + np.sqrt(9 - 8*(1.0 - len(eigenvector))))/4).astype(int)
    lup = lmax
    ldown = -lmax
    nmax = lmax

    Rl_table = R_computation_v2(eigenvector, ldown, lup, nmax, r, a, b, SPS.taub)

    max_freq = 5e9
    n_signals = 10

    freq, lambda1tot, time, list_signals = headtail_signal_plot(Rl_table, max_freq=max_freq,
                                                                 nb_points=300, lup=lup, ldown=ldown,
                                                                 eigenvalue=eigenvalue, r=r, tune=SPS.Qx,
                                                                 Qp=0, eta=SPS._eta, radius=SPS.radius, beta=SPS.beta,
                                                                 n_signals=n_signals)

    import matplotlib.pyplot as plt

    #First plot shows the spectrum
    plt.figure()
    plt.xlim([-max_freq, max_freq])
    plt.plot(freq, np.real(lambda1tot), '-r')
    plt.plot(freq, np.imag(lambda1tot), '-b')
    plt.xlabel('Frequency / Hz')

    #Second plot shows a serie of signals
    plt.figure()
    for ii in np.arange(0, n_signals, 1):
        plt.plot(time, list_signals[ii], '-')

    plt.xlim(-6 * SPS.taub/4, 6 * SPS.taub/4)
    plt.xlabel('Bunch length / ns')
    plt.ylabel('Signal amplitude / arb. unit')
    '''


    c = 299792458
    omega = 2 * np.pi * np.linspace(-max_freq, max_freq, nb_points)

    lambda1 = np.zeros(len(omega), dtype=complex)
    lambda1tot = np.zeros(len(omega), dtype=complex)

    for i in range(0, len(omega)):
        for l in range(lup, ldown-1, -1):
            lambda1[i] = np.trapz(r*Rl_table[l-ldown]*2*np.pi*(-1.0j)**(-l)
                                  *scipy.special.jn(l,(eigenvalue + omega[i])
                                                    *r/(beta*c) - Qp*r/(eta * radius)), r)
            lambda1tot[i] += lambda1[i]


    list_signals = []
    for ii in np.arange(0, n_signals, 1):
        # sig_lambda = lambda1tot*np.exp(ii*1j*2*np.pi*(machine.Qxfrac+eigenvalues_converged[Qpindex][EValue_number]/omega0))
        # signal = np.fft.ifft(sig_lambda+np.conj(sig_lambda))
        signal = np.fft.ifft(np.fft.fftshift(lambda1tot))*np.exp(ii*1j*2*np.pi*tune)
        # signal = np.fft.ifft(np.fft.fftshift(lambda1tot))
        # signal = signal[0::2]

        #freq = np.fft.fftfreq(tau.shape[-1])

        n = signal.size
        time = np.fft.fftfreq(n, d=np.diff(omega/2/np.pi)[0])
        ind = np.argsort(time)
        list_signals.append((np.abs(signal) * np.cos(np.angle(signal)))[ind])

    return omega/(2*np.pi), lambda1tot, time[ind], list_signals

sigma_b = MM_obj.sigma_b
r_vect = np.linspace(0., 3*sigma_b, 1000)
Rl_table = R_computation_v2(
        evect[:,:,i_most_unstable].reshape(
            len(MM_obj.l_vect)*len(MM_obj.m_vect)),
        MM_obj.l_min, MM_obj.l_max,
        MM_obj.m_max,
        r_vect, 8.,
        8., 4*sigma_b/clight,
        beta=1)

freq, lambda1tot, times, list_signals = headtail_signal(
        Rl_table, 30e9, 1000, MM_obj.l_max, MM_obj.l_min,
                Omega_mat[i_most_unstable],
                r_vect, MM_obj.Q_full,
                0., MM_obj.eta,
                27e3/(2*np.pi),
                beta=1, n_signals=21)

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
for ss in list_signals:
    ax1.plot(times, ss)
plt.show()
