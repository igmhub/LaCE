import numpy as np
import copy


class PolyP1D(object):
    """Log-polynomial representation of simulation P1D in comoving units."""

    def __init__(
        self,
        k_Mpc=None,
        P_Mpc=None,
        lnP_fit=None,
        kmin_Mpc=1.0e-3,
        kmax_Mpc=10.0,
        deg=4,
    ):
        """Fit measured P1D or reconstruct a log-polynomial representation.

        Parameters
        ----------
        k_Mpc, P_Mpc : array-like, optional
            Measured positive wavenumbers in 1/Mpc and P1D values in Mpc.
            Supplying ``k_Mpc`` selects a new least-squares fit.
        lnP_fit : array-like, optional
            Polynomial coefficients in ``ln(k_Mpc)`` used when no measurements
            are supplied.
        kmin_Mpc, kmax_Mpc : float, default=1e-3, 10
            Open fit interval in 1/Mpc.
        deg : int, default=4
            Polynomial degree for a measured-power fit.
        """

        if k_Mpc is None:
            self._setup_from_coefficients(lnP_fit, kmin_Mpc)
        else:
            self._setup_from_measured(k_Mpc, P_Mpc, kmin_Mpc, kmax_Mpc, deg)

    def _setup_from_measured(self, k_Mpc, P_Mpc, kmin_Mpc, kmax_Mpc, deg):
        """Fit logarithmic P1D over the selected comoving k interval."""

        # we need to mask k=0 and high-k (or will dominate fit)
        kfit = (k_Mpc < kmax_Mpc) & (k_Mpc > kmin_Mpc)
        self.lnP_fit = np.polyfit(np.log(k_Mpc[kfit]), np.log(P_Mpc[kfit]), deg)
        # store poly1d object
        self.lnP = np.poly1d(self.lnP_fit)
        # remember minimum k used in fit (better not to extrapolate)
        self.kmin_Mpc = min(k_Mpc[kfit])

    def _setup_from_coefficients(self, lnP_fit, kmin_Mpc):
        """Initialize the polynomial directly from logarithmic coefficients."""

        # store poly1d object
        self.lnP = np.poly1d(lnP_fit)
        # remember minimum k used in fit (better not to extrapolate)
        self.kmin_Mpc = kmin_Mpc

    def P_Mpc(self, k_Mpc):
        """Evaluate smooth P1D in Mpc on comoving input wavenumbers.

        Values below the fitted lower boundary are evaluated at ``kmin_Mpc``
        rather than extrapolated.
        """

        # do not extrapolate below minimum k used in fit
        k = copy.copy(k_Mpc)
        k[k_Mpc < self.kmin_Mpc] = self.kmin_Mpc
        return np.exp(self.lnP(np.log(k)))


def fit_polynomial(xmin, xmax, x, y, deg=2):
    """Fit ``ln(y)`` as a polynomial of ``ln(x)`` inside an open interval.

    Parameters
    ----------
    xmin, xmax : float
        Open fitting interval in the units of ``x``.
    x, y : array-like
        Positive independent and dependent values.
    deg : int, default=2
        Polynomial degree.

    Returns
    -------
    numpy.poly1d
        Polynomial evaluated on ``ln(x)``.
    """
    x_fit = (x > xmin) & (x < xmax)
    # We could make these less correlated by better choice of parameters
    poly = np.polyfit(np.log(x[x_fit]), np.log(y[x_fit]), deg=deg)
    return np.poly1d(poly)
