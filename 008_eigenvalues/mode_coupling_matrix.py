import numpy as np
from scipy.special import assoc_laguerre, gamma
from scipy.constants import c as clight
from numpy.linalg import eigvals
from multiprocessing import Pool

def compute_R_tilde_for_one_l(
        i_l, ll, n_m, n_r, n_n, m_vect, i_l_zero, n_l_pos,
        e_L_PHI_mat, r_vect, phi_vect,
        r_b, sigma_b, a_param, dr, dphi,
        cos_phi, cos2_phi, z_slices,
        HH, H_N_2_vect, exp_j_dPhi_R_PHI):

    r_part_l_M_R_mat = np.zeros((n_m, n_r))
    R_plus_curr = np.zeros((n_m, n_n), dtype=np.complex)
    R_minus_curr = np.zeros((n_m, n_n), dtype=np.complex)
    for i_m, mm in  enumerate(m_vect):
        print(f'l={i_l-i_l_zero}/{n_l_pos} m={i_m}/{n_m}')
        lag_l_m_R_vect =assoc_laguerre(
                a_param * r_vect*r_vect, n=mm, k=np.abs(ll))
        r_part_l_M_R_mat[i_m, :]  = (
                  dr * r_vect
                * (r_vect/r_b)**np.abs(ll)
                * lag_l_m_R_vect
                )
        for nn in range(n_n):
            int_dphi_l_n_R_vect = np.zeros(n_r, dtype=np.complex)
            int_dphi_ml_n_R_vect = np.zeros(n_r, dtype=np.complex)
            for i_r, rr in enumerate(r_vect):
                h_n_r_cos_phi = np.interp(rr*cos_phi,
                    z_slices, HH[nn, :])
                exp_c2_r_PHI_vect = np.exp(-a_param*rr*rr
                        *(1-cos2_phi/(2*a_param*sigma_b**2)))
                integrand_part = (exp_c2_r_PHI_vect
                  * h_n_r_cos_phi/H_N_2_vect[nn]
                  * np.conj(e_L_PHI_mat[i_l, :]))
                int_dphi_l_n_R_vect[i_r] = dphi * np.sum(
                    integrand_part
                  * np.conj(exp_j_dPhi_R_PHI[i_r, :]))
                int_dphi_ml_n_R_vect[i_r] = dphi * np.sum(
                    np.conj(integrand_part)
                  * np.conj(exp_j_dPhi_R_PHI[i_r, :]))
            R_plus_curr[i_m, nn] = np.sum(
                    r_part_l_M_R_mat[i_m, :]*
                    int_dphi_l_n_R_vect)
            R_minus_curr[i_m, nn] = np.sum(
                    r_part_l_M_R_mat[i_m, :]*
                    int_dphi_ml_n_R_vect)
    return R_plus_curr, R_minus_curr

def compute_R_for_one_l(
        i_l, ll, n_m, n_r, n_n, m_vect, i_l_zero, n_l_pos,
        e_L_PHI_mat, r_vect, phi_vect,
        r_b, sigma_b, a_param, dr, dphi,
        cos_phi, z_slices, KK, exp_j_dPhi_R_PHI):

    r_part_l_M_R_mat = np.zeros((n_m, n_r))
    R_plus_curr = np.zeros((n_m, n_n), dtype=np.complex)
    R_minus_curr = np.zeros((n_m, n_n), dtype=np.complex)
    for i_m, mm in  enumerate(m_vect):
        print(f'l={i_l-i_l_zero}/{n_l_pos} m={i_m}/{n_m}')
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
            int_dphi_ml_n_R_vect = np.zeros(n_r, dtype=np.complex)
            for i_r, rr in enumerate(r_vect):
                k_n_r_cos_phi = np.interp(rr*cos_phi,
                    z_slices, KK[nn, :])
                integrand_part = k_n_r_cos_phi*e_L_PHI_mat[i_l, :]
                int_dphi_l_n_R_vect[i_r] = dphi * np.sum(
                    exp_j_dPhi_R_PHI[i_r, :]
                  * integrand_part)
                int_dphi_ml_n_R_vect[i_r] = dphi * np.sum(
                    exp_j_dPhi_R_PHI[i_r, :]
                  * np.conj(integrand_part))
            R_plus_curr[i_m, nn] = np.sum(
                    r_part_l_M_R_mat[i_m, :]*
                    int_dphi_l_n_R_vect)
            R_minus_curr[i_m, nn] = np.sum(
                    r_part_l_M_R_mat[i_m, :]*
                    int_dphi_ml_n_R_vect)
    return R_plus_curr, R_minus_curr

def f_R_tilde_for_pool(args):
    return  compute_R_tilde_for_one_l(*args)

def f_R_for_pool(args):
    return  compute_R_for_one_l(*args)

class CouplingMatrix(object):

    def __init__(self, z_slices, HH, KK, l_min,
            l_max, m_max, n_phi, n_r, N_max, Q_full, sigma_b, r_b,
            a_param, omega0, omega_s, eta=None, alpha_p=(), beta_p=(),
            R_tilde_lmn=None, R_lmn=None, MM = None, beta_fun_rescale=None,
            pool_size=0):

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
        self.omega0 = omega0
        self.omega_s = omega_s
        self.eta = eta
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.beta_fun_rescale = beta_fun_rescale

        l_vect = np.array(range(l_min, l_max+1))
        m_vect = np.array(range(0, m_max+1))

        self.l_vect = l_vect
        self.m_vect = m_vect

        if MM is None:
            assert(l_min == -l_max)
            r_max = np.max(np.abs(z_slices))
            dz = z_slices[1] - z_slices[0]

            r_vect = np.linspace(0, r_max, n_r)
            phi_vect = np.linspace(0, 2*np.pi, n_phi+1)[:-1]

            dphi = phi_vect[1] - phi_vect[0]
            dr = r_vect[1] - r_vect[0]

            sin_phi = np.sin(phi_vect)
            cos_phi = np.cos(phi_vect)
            cos2_phi = cos_phi*cos_phi

            radius = clight/omega0
            beta_fun_smooth = radius/Q_full
            if beta_fun_rescale is None:
                beta_fun_rescale = beta_fun_smooth

            # Compute phase shift term
            dPhi_R_PHI = np.zeros((n_r, n_phi))
            if len(alpha_p) > 0:
                aP_terms = len(alpha_p)
                A_P = -beta_fun_rescale * alpha_p/4/ np.pi

                C_N_PHI = np.zeros((aP_terms, n_phi))
                C_bar_N = np.zeros(aP_terms)
                for nn in range(aP_terms):
                    if nn == 0:
                        C_bar_N[nn] = 2*np.pi
                        C_N_PHI[nn, :] = phi_vect
                        continue
                    if nn == 1:
                        C_bar_N[nn] = 0.
                        C_N_PHI[nn, :] = sin_phi
                        continue
                    C_bar_N[nn] = (nn-1.)/nn * C_bar_N[nn-2]
                    C_N_PHI[nn, :] = (cos_phi**(nn-1)*sin_phi/nn
                            + (nn-1.)/nn * C_N_PHI[nn-2, :])

                for nn in range(aP_terms):
                    dPhi_R_PHI += -omega0/omega_s * A_P[nn] * np.dot(
                            np.atleast_2d(r_vect**nn).T,
                            np.atleast_2d(C_N_PHI[nn, :]
                             - C_bar_N[nn]/(2*np.pi)*phi_vect))
            if len(beta_p) > 0:
                bP_terms = len(beta_p)
                B_P = beta_p

                S_N_PHI = np.zeros((bP_terms, n_phi))
                S_bar_N = np.zeros(bP_terms)
                for nn in range(bP_terms):
                    if nn == 0:
                        S_bar_N[nn] = 2*np.pi
                        S_N_PHI[nn, :] = phi_vect
                        continue
                    if nn == 1:
                        S_bar_N[nn] = 0.
                        S_N_PHI[nn, :] = -cos_phi
                        continue
                    S_bar_N[nn] = (nn-1.)/nn * S_bar_N[nn-2]
                    S_N_PHI[nn, :] = -((sin_phi**(nn-1)*cos_phi/nn)
                            + (nn-1)/nn * S_N_PHI[nn-2, :])
                for nn in range(bP_terms):
                    dPhi_R_PHI += -omega0/omega_s * B_P[nn] \
                            * (omega_s/(clight*eta))**nn * np.dot(
                            np.atleast_2d(r_vect**nn).T,
                            np.atleast_2d(S_N_PHI[nn, :]
                                - S_bar_N[nn]/(2*np.pi)*phi_vect))

            exp_j_dPhi_R_PHI = np.exp(1j*dPhi_R_PHI)
            # For checks:
            self.dPhi_R_PHI= dPhi_R_PHI[:, :]
            self.d_Q_R_PHI = -omega_s/omega0 * np.diff(dPhi_R_PHI[:, :], axis=1)/dphi
            # End phase shift 

            l_vect = np.array(range(l_min, l_max+1))
            m_vect = np.array(range(0, m_max+1))

            n_l = len(l_vect)
            n_m = len(m_vect)
            n_n = N_max + 1

            n_l_pos = np.sum(np.int_(l_vect>=0))
            i_l_zero = np.where(l_vect==0)[0][0]

            KK[np.isnan(KK)] = 0
            KK *= beta_fun_rescale/beta_fun_smooth

            H_N_2_vect = dz * np.sum(HH**2, axis=1)

            e_L_PHI_mat = np.zeros((n_l, n_phi), dtype=np.complex)
            for i_l, ll in enumerate(l_vect):
                e_L_PHI_mat[i_l, :] = np.exp(1j*ll*phi_vect)

            # Remember that Ks and Hs do not have the last point at 360 deg

            # Compute R_tilde integrals
            print('Compute R_tilde_lmn ...')
            R_tilde_lmn = np.zeros((n_l, n_m, n_n), dtype=np.complex)
            if pool_size == 0:
                print('Serial implementation')
                for i_l, ll in enumerate(l_vect):
                    if ll<0:
                        continue
                    R_plus_curr, R_minus_curr = compute_R_tilde_for_one_l(
                        i_l, ll, n_m, n_r, n_n, m_vect, i_l_zero, n_l_pos,
                        e_L_PHI_mat, r_vect, phi_vect,
                        r_b, sigma_b, a_param, dr, dphi,
                        cos_phi, cos2_phi, z_slices, HH, H_N_2_vect,
                        exp_j_dPhi_R_PHI)
                    R_tilde_lmn[i_l, :, :] = R_plus_curr
                    i_mlp = np.where(l_vect==-ll)[0][0]
                    R_tilde_lmn[i_mlp, :, :] = R_minus_curr
            else:
                print('Parallel implementation')
                n_pools = len(l_vect)/pool_size
                if pool_size>1:
                    pool = Pool(processes=pool_size)
                other_args= [n_m, n_r, n_n, m_vect, i_l_zero, n_l_pos,
                    e_L_PHI_mat, r_vect, phi_vect,
                    r_b, sigma_b, a_param, dr, dphi,
                    cos_phi, cos2_phi, z_slices, HH, H_N_2_vect,
                    exp_j_dPhi_R_PHI]
                i_l = 0
                while (i_l<n_l):
                    ll = l_vect[i_l]
                    if ll < 0:
                        i_l +=1
                        continue
                    i_l_pool = [iii for iii in range(i_l, i_l+pool_size) if iii<n_l]
                    args_pool = [[iii, l_vect[iii]]+other_args
                            for iii in i_l_pool if iii<n_l]
                    R_pool = pool.map(f_R_tilde_for_pool, args_pool)

                    for ilp, Rp in zip(i_l_pool, R_pool):
                        llp = l_vect[ilp]
                        R_tilde_lmn[ilp, :, :] = Rp[0]
                        i_mlp = np.where(l_vect==-llp)[0][0]
                        R_tilde_lmn[i_mlp, :, :] = Rp[1]
                    i_l += len(i_l_pool)

            # Compute R integrals
            print('Compute R_lmn ...')
            R_lmn = np.zeros((n_l, n_m, n_n), dtype=np.complex)
            if pool_size == 0:
                print('Serial implementation')
                for i_l, ll in enumerate(l_vect):
                    if ll<0:
                        continue
                    R_plus_curr, R_minus_curr = compute_R_for_one_l(
                        i_l, ll, n_m, n_r, n_n, m_vect, i_l_zero, n_l_pos,
                        e_L_PHI_mat, r_vect, phi_vect,
                        r_b, sigma_b, a_param, dr, dphi,
                        cos_phi, z_slices, KK, exp_j_dPhi_R_PHI)
                    R_lmn[i_l, :, :] = R_plus_curr
                    i_ml = np.where(l_vect==-ll)[0][0]
                    R_lmn[i_ml, :, :] =  R_minus_curr
            else:
                print('Parallel implementation')
                n_pools = len(l_vect)/pool_size
                if pool_size>1:
                    pool = Pool(processes=pool_size)
                other_args= [n_m, n_r, n_n, m_vect, i_l_zero, n_l_pos,
                        e_L_PHI_mat, r_vect, phi_vect,
                        r_b, sigma_b, a_param, dr, dphi,
                        cos_phi, z_slices, KK, exp_j_dPhi_R_PHI]
                i_l = 0
                while (i_l<n_l):
                    ll = l_vect[i_l]
                    if ll < 0:
                        i_l +=1
                        continue
                    i_l_pool = [iii for iii in range(i_l, i_l+pool_size) if iii<n_l]
                    args_pool = [[iii, l_vect[iii]]+other_args
                            for iii in i_l_pool if iii<n_l]
                    R_pool = pool.map(f_R_for_pool, args_pool)
                    for ilp, Rp in zip(i_l_pool, R_pool):
                        llp = l_vect[ilp]
                        R_lmn[ilp, :, :] = Rp[0]
                        i_mlp = np.where(l_vect==-llp)[0][0]
                        R_lmn[i_mlp, :, :] = Rp[1]
                    i_l += len(i_l_pool)

            self.beta_fun_rescale = beta_fun_rescale
            self.beta_fun_smooth = beta_fun_smooth
            self.R_tilde_lmn = R_tilde_lmn
            self.R_lmn = R_lmn
            self.MM = self.compute_final_matrix()
            self.r_vect = r_vect
            self.phi_vect = phi_vect
        else:
            self.R_tilde_lmn = R_tilde_lmn
            self.R_lmn = R_lmn
            self.MM = MM

    def compute_final_matrix(self, N_max_cut=None):

        n_l = len(self.l_vect)
        n_m = len(self.m_vect)

        if N_max_cut is not None:
            assert(N_max_cut <= self.N_max)
            n_cut = N_max_cut
        else:
            n_cut = self.N_max

        print('Compute final matrix')
        no_coeff_M_l_m_lp_mp = np.zeros((n_l, n_m, n_l, n_m), dtype=np.complex)
        for i_l, ll in enumerate(self.l_vect):
            for i_m, mm in enumerate(self.m_vect):
                for i_lp in range(n_l):
                    for i_mp in range(n_m):
                        temp = gamma(mm + 1) / gamma(np.abs(ll) + mm + 1)

                        no_coeff_M_l_m_lp_mp[i_l, i_m, i_lp, i_mp] = (
                                temp * np.sum(self.R_tilde_lmn[i_lp, i_mp, :][:n_cut]
                                    * self.R_lmn[i_l, i_m, :][:n_cut]))

        coeff = -clight*self.a_param/(4*np.pi**2*np.sqrt(2*np.pi)*self.Q_full*self.sigma_b)
        MM = coeff*no_coeff_M_l_m_lp_mp

        return MM

    def get_sub_matrix(self, l_min, l_max, m_max, N_max=None):

        assert(l_min >= self.l_min)
        assert(l_max <= self.l_max)
        assert(m_max <= self.m_max)

        mask_m_keep = self.m_vect<=m_max
        mask_l_keep = (self.l_vect<=l_max) & (self.l_vect>=l_min)

        new_MM = self.compute_final_matrix(N_max_cut=N_max)

        new_MM = new_MM[mask_l_keep, :, :, :]
        new_MM = new_MM[:, mask_m_keep, :, :]
        new_MM = new_MM[:, :, mask_l_keep, :]
        new_MM = new_MM[:, :, :, mask_m_keep]

        # Patch to load old pickles
        if not hasattr(self, 'eta'):
            self.eta = None
        if not hasattr(self, 'beta_p'):
            self.beta_p = ()

        new = CouplingMatrix(
            z_slices=self.z_slices,
            HH=self.HH,
            KK=self.KK,
            l_min=l_min,
            l_max=l_max,
            m_max=m_max,
            n_phi=self.n_phi,
            n_r=self.n_r,
            N_max=self.N_max,
            Q_full=self.Q_full,
            sigma_b=self.sigma_b,
            r_b=self.r_b,
            a_param=self.a_param,
            omega0 = self.omega0,
            omega_s = self.omega_s,
            eta=self.eta,
            alpha_p = self.alpha_p,
            beta_p = self.beta_p,
            MM = new_MM,
            beta_fun_rescale=self.beta_fun_rescale)

        return new

    def compute_mode_complex_freq(self, omega_s, rescale_by=np.array([1.])):

        Omega_mat = []
        n_l = len(self.l_vect)
        n_m = len(self.m_vect)
        for ii, rr in enumerate(rescale_by):
            print(f'{ii}/{len(rescale_by)}')
            MM_m_l_omegas = self.MM.copy()
            MM_m_l_omegas *= rr

            for i_l, ll in enumerate(self.l_vect):
                for i_m, mm in enumerate(self.m_vect):
                    for i_lp, llp in enumerate(self.l_vect):
                        for i_mp, mm_p in enumerate(self.m_vect):
                            if i_l == i_lp and i_m == i_mp:
                                MM_m_l_omegas[i_l, i_m, i_lp, i_mp] += ll*omega_s
            mat_to_diag = MM_m_l_omegas.reshape((n_l*n_m,
                                    n_l*n_m))
            Omega_mat.append(eigvals(mat_to_diag))

        return np.squeeze(np.array(Omega_mat))
