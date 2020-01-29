import numpy as np

from PyHEADTAIL.general.element import Element

class ResponseMatrix(Element):

    def __init__(self, slicer=None, response_data_file=None, coord=None, kick_factor=1.,
            n_terms_to_be_kept=None, n_tail_cut=0):

        if coord != 'x':
            raise ValueError('Only x plane implemented for now!')

        response_data_file = 'response_data.mat'
        ob_responses = mfm.myloadmat_to_obj(response_data_file)

        z_resp = ob_responses.z_slices

        # Clean-up NaNs
        x_resp_mat = ob_responses.x_mat
        x_resp_mat[np.isnan(x_resp_mat)] = 0.
        dpx_resp_mat = ob_responses.dpx_mat
        dpx_resp_mat[np.isnan(dpx_resp_mat)] = 0.

        # Prepare submatrices
        FF = x_resp_mat[:, :].T    # base functions are in the columns
        MM = dpx_resp_mat[:, :].T  # response to base functions are in the columns
        RR = np.dot(FF.T, FF)      # Matrix of cross products
        RR_inv = np.linalg.inv(RR) # Its inverse

        # Prepare cut on the harmonics
        if n_terms_to_be_kept is not None:
            self.n_terms_to_be_kept = n_terms_to_be_kept
        else:
            self.n_terms_to_be_kept = len(z_resp)
        CC = 0*MM
        for ii in range(self.n_terms_to_be_kept):
            CC[ii, ii] = 1

        # Prepare cut for the tails
        CC_tails = np.identity(len(z_resp))
        for ii in range(n_tail_cut):
            CC_tails[ii, ii] = 0.
            CC_tails[-ii, -ii] = 0.

        # Build response matrix
        WW_no_harmonic_cut = np.dot(MM, np.dot(RR_inv, FF.T))
        WW = np.dot(MM, np.dot(CC, np.dot(RR_inv, np.dot(FF.T, CC_tails))))

        # Bind matrices
        self.z_resp = z_resp
        self.FF = FF
        self.MM = MM
        self.RR = RR
        self.RR_inv = RR_inv
        self.CC = CC
        self.CC_tails
        self.WW_no_harmonic_cut = WW_no_harmonic_cut
        self.WW = WW

    def response_to_slice_array(self, arr):
        return np.dot(self.WW, arr.T)

    def track(self, bunch):
        slices = bunch.get_slices(
                self.slicer, statistics=['mean_'+ self.coord])
        arr = getattr(slices, 'mean_'+ self.coord)
        dp_slices = np.dot(self.WW, arr.T)
        dp_prt = np.interp(bunch.z, slices.z_centers, dp_slices)
        p_old = getattr(bunch, self.coord + 'p')
        setattr(bunch, self.coord + 'p', p_old + dp_prt)

