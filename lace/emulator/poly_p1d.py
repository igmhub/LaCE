import numpy as np
import copy


class PolyP1D(object):
    """Polynomial describing P1D measured in a simulation."""

    def __init__(
        self,
        k_Mpc=None,
        P_Mpc=None,
        lnP_fit=None,
        kmin_Mpc=1.0e-3,
        kmax_Mpc=10.0,
        deg=4,
    ):
        """Setup object either by passing measured power, or coefficients"""

        if k_Mpc is None:
            self._setup_from_coefficients(lnP_fit, kmin_Mpc)
        else:
            self._setup_from_measured(k_Mpc, P_Mpc, kmin_Mpc, kmax_Mpc, deg)

    def _setup_from_measured(self, k_Mpc, P_Mpc, kmin_Mpc, kmax_Mpc, deg):
        """Fit input power and store poly1d object"""

        # we need to mask k=0 and high-k (or will dominate fit)
        kfit = (k_Mpc < kmax_Mpc) & (k_Mpc > kmin_Mpc)
        self.lnP_fit = np.polyfit(np.log(k_Mpc[kfit]), np.log(P_Mpc[kfit]), deg)
        # store poly1d object
        self.lnP = np.poly1d(self.lnP_fit)
        # remember minimum k used in fit (better not to extrapolate)
        self.kmin_Mpc = min(k_Mpc[kfit])

    def _setup_from_coefficients(self, lnP_fit, kmin_Mpc):
        """Setup object from coefficients"""

        # store poly1d object
        self.lnP = np.poly1d(lnP_fit)
        # remember minimum k used in fit (better not to extrapolate)
        self.kmin_Mpc = kmin_Mpc

    def P_Mpc(self, k_Mpc):
        """Evaluate smooth power at input array k (in Mpc)"""

        # do not extrapolate below minimum k used in fit
        k = copy.copy(k_Mpc)
        k[k_Mpc < self.kmin_Mpc] = self.kmin_Mpc
        return np.exp(self.lnP(np.log(k)))
