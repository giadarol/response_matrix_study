import numpy as np
from scipy.special import assoc_laguerre, gamma
from scipy.constants import c as clight
from numpy.linalg import eigvals

class CouplingMatrix(object):

    def __init__(self, z_slices, HH, KK, l_min,
            l_max, m_max, n_phi, n_r, N_max, Q_full, sigma_b, r_b,
            a_param):

        self.z_slices = z_slices
        self.HH       = HH
        self.KK       = KK
        self.l_min    = l_min
        self.l_max    = l_max
        self.m_max    = m_max
        self.n_phi    = n_phi
        self.n_r      = n_r
        self.N_max    = N_max
        self.Q_full   = Q_full
        self.sigma_b  = sigma_b
        self.r_b      = r_b
        self.a_param  = a_param

        r_max = np.max(np.abs(z_slices))
        dz = z_slices[1] - z_slices[0]

        r_vect = np.linspace(0, r_max, n_r)
        phi_vect = np.linspace(0, 2*np.pi, n_phi)

        dphi = phi_vect[1] - phi_vect[0]
        dr = r_vect[1] - r_vect[0]

        l_vect = np.array(range(l_min, l_max+1))
        m_vect = np.array(range(0, m_max+1))

        n_l = len(l_vect)
        n_m = len(m_vect)
        n_n = N_max + 1

        KK[np.isnan(KK)] = 0

        H_N_2_vect = dz * np.sum(HH**2, axis=1)

        cos_phi = np.cos(phi_vect)
        cos2_phi = cos_phi*cos_phi

        e_L_PHI_mat = np.zeros((n_l, n_phi), dtype=np.complex)
        for i_l, ll in enumerate(l_vect):
            e_L_PHI_mat[i_l, :] = np.exp(1j*ll*phi_vect)

        # Remember that Ks and Hs do not have the last point at 360 deg
        # Compute R_tilde integrals
        print('Compute R_tilde_lmn ...')
        R_tilde_lmn = np.zeros((n_l, n_m, n_n), dtype=np.complex)
        for i_l, ll in enumerate(l_vect):
            print(f'{i_l}/{n_l}')
            r_part_l_M_R_mat = np.zeros((n_m, n_r))
            for i_m, mm in  enumerate(m_vect):
                lag_l_m_R_vect =assoc_laguerre(
                        a_param * r_vect*r_vect, n=mm, k=np.abs(ll))
                r_part_l_M_R_mat[i_m, :]  = (
                          dr * r_vect
                        * (r_vect/r_b)**np.abs(ll)
                        * lag_l_m_R_vect
                        )
                for nn in range(n_n):
                    int_dphi_l_n_R_vect = np.zeros(n_r, dtype=np.complex)
                    for i_r, rr in enumerate(r_vect):
                        h_n_r_cos_phi = np.interp(rr*cos_phi,
                            z_slices, HH[nn, :])
                        exp_c2_r_PHI_vect = np.exp(-a_param*rr*rr
                                *(1-cos2_phi/(2*a_param*sigma_b**2)))
                        int_dphi_l_n_R_vect[i_r] = dphi * np.sum(
                            exp_c2_r_PHI_vect
                          * h_n_r_cos_phi/H_N_2_vect[nn]
                          * np.conj(e_L_PHI_mat[i_l, :]))

                    R_tilde_lmn[i_l, i_m, nn] = np.sum(
                            r_part_l_M_R_mat[i_m, :]*
                            int_dphi_l_n_R_vect)

        # Compute R integrals
        print('Compute R_lmn ...')
        R_lmn = np.zeros((n_l, n_m, n_n), dtype=np.complex)
        for i_l, ll in enumerate(l_vect):
            print(f'{i_l}/{n_l}')
            r_part_l_M_R_mat = np.zeros((n_m, n_r))
            for i_m, mm in  enumerate(m_vect):
                lag_l_m_R_vect =assoc_laguerre(
                        a_param * r_vect*r_vect, n=mm, k=np.abs(ll))
                r_part_l_M_R_mat[i_m, :]  = (
                          dr * r_vect
                        * (a_param*r_b*r_vect)**np.abs(ll)
                        * lag_l_m_R_vect
                        * np.exp(-r_vect**2 / (2*sigma_b**2))
                        )

                for nn in range(n_n):
                    int_dphi_l_n_R_vect = np.zeros(n_r, dtype=np.complex)
                    for i_r, rr in enumerate(r_vect):
                        k_n_r_cos_phi = np.interp(rr*cos_phi,
                            z_slices, KK[nn, :])
                        int_dphi_l_n_R_vect[i_r] = dphi * np.sum(
                            k_n_r_cos_phi*e_L_PHI_mat[i_l, :])

                    R_lmn[i_l, i_m, nn] = np.sum(
                            r_part_l_M_R_mat[i_m, :]*
                            int_dphi_l_n_R_vect)

        print('Compute final matrix')
        no_coeff_M_l_m_lp_mp = np.zeros((n_l, n_m, n_l, n_m), dtype=np.complex)
        for i_l, ll in enumerate(l_vect):
            for i_m, mm in enumerate(m_vect):
                for i_lp in range(n_l):
                    for i_mp in range(n_m):
                        temp = gamma(mm + 1) / gamma(np.abs(ll) + mm + 1)

        #                # To be handled:
        #                assert(not(np.isnan(temp)))
        #                assert(not(np.isinf(temp)))

                        no_coeff_M_l_m_lp_mp[i_l, i_m, i_lp, i_mp] = (
                                temp * np.sum(R_tilde_lmn[i_lp, i_mp, :]
                                    * R_lmn[i_l, i_m, :]))

        coeff = -clight*a_param/(4*np.pi**2*np.sqrt(2*np.pi)*Q_full*sigma_b)
        MM = coeff*no_coeff_M_l_m_lp_mp

        self.l_vect = l_vect
        self.m_vect = m_vect
        self.n_l = n_l
        self.n_m = n_m
        self.MM = MM


    def compute_mode_complex_freq(self, omega_s, rescale_by=np.array([1.])):

        Omega_mat = []
        for ii, rr in enumerate(rescale_by):

            MM_m_l_omegas = self.MM.copy()
            MM_m_l_omegas *= rr

            for i_l, ll in enumerate(self.l_vect):
                for i_m, mm in enumerate(self.m_vect):
                    for i_lp, llp in enumerate(self.l_vect):
                        for i_mp, mm_p in enumerate(self.m_vect):
                            if i_l == i_lp and i_m == i_mp:
                                MM_m_l_omegas[i_l, i_m, i_lp, i_mp] += ll*omega_s
            mat_to_diag = MM_m_l_omegas.reshape((self.n_l*self.n_m,
                                    self.n_l*self.n_m))
            Omega_mat.append(eigvals(mat_to_diag))

        return np.squeeze(np.array(Omega_mat))
