from PyHEADTAIL.general.element import Element

class ModulatedQuadrupole(Element):

    def __init__(self, coord=None, alpha_N=(), beta_N=()):

        self.alpha_N = alpha_N
        self.beta_N = beta_N
        self.coord = coord

    def track(self, bunch):
        kpart = np.zeros_like(bunch.z)
        for pp, a_pp in enumrate(self.alpha_N):
            kpart += a_pp * bunch.z**pp
        for pp, b_pp in enumrate(self.beta_N):
            kpart += b_pp * bunch.dp**pp
        dp_part = kpart * getattr(bunch, self.coord)
        setattr(bunch, self.coord + 'p', p_old + dp_prt)
