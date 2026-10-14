import numpy as np


class Nonlinear_Smoothing(object):
    """Smooth one-dimensional flux-power spectra in log wavenumber space."""

    def __init__(
        self,
        data_set_kernel,
        kmax_Mpc,
        bandwidth=[0.8, 0.4, 0.2],
        krange=[0.15, 1, 2.5, 4],
    ):
        """Fit piecewise kernel smoothers from a reference P1D data set.

        Parameters
        ----------
        data_set_kernel : sequence of mapping
            Reference rows containing positive ``k_Mpc`` and ``p1d_Mpc`` arrays.
        kmax_Mpc : float
            Largest wavenumber retained for smoother construction, in 1/Mpc.
        bandwidth : sequence of float
            Kernel bandwidths, one for each wavenumber interval.
        krange : sequence of float
            Boundaries in 1/Mpc defining piecewise smoothing intervals.
        """

        self.bandwidth = bandwidth
        self.krange = krange
        self.kmax_Mpc = kmax_Mpc

        log_data = self._interp_for_smoothing(data_set_kernel)
        self._set_kernel_smoothing(log_data)

    def _interp_for_smoothing(self, data):
        """Interpolate positive P1D rows onto a common logarithmic k grid.

        Returns
        -------
        ndarray
            Logarithmic P1D values with shape ``(nspectrum, ninterpolated_k)``.
        """

        mask = np.argwhere(
            (data[0]["k_Mpc"] > 0) & (data[0]["k_Mpc"] < self.kmax_Mpc)
        )[:, 0]
        logk_Mpc = np.log(data[0]["k_Mpc"][mask])

        self.interp_logk_Mpc = np.linspace(
            logk_Mpc[0], logk_Mpc[-1], logk_Mpc.shape[0] * 2
        )
        log_data = np.zeros((len(data), self.interp_logk_Mpc.shape[0]))
        for isim in range(len(data)):
            log_data[isim] = np.interp(
                self.interp_logk_Mpc,
                logk_Mpc,
                np.log(data[isim]["p1d_Mpc"][mask]),
            )
        return log_data

    def _set_kernel_smoothing(self, log_data):
        """Fit one Nadaraya-Watson smoother per configured k interval."""
        import skfda
        from skfda.preprocessing.smoothing import KernelSmoother
        from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
        from skfda.misc.kernels import epanechnikov

        dat = skfda.FDataGrid(log_data, grid_points=self.interp_logk_Mpc)
        self.kernel = []
        for ii in range(len(self.bandwidth)):
            _ = KernelSmoother(
                kernel_estimator=NadarayaWatsonHatMatrix(
                    bandwidth=self.bandwidth[ii], kernel=epanechnikov
                ),
            )
            self.kernel.append(_.fit(dat))

    def apply_kernel_smoothing(self, k_Mpc, data):
        """Apply piecewise smoothing and return P1D on a requested k grid.

        Parameters
        ----------
        k_Mpc : ndarray
            Positive output wavenumber grid in 1/Mpc.
        data : mapping or list of mapping
            One or more rows containing ``k_Mpc`` and ``p1d_Mpc`` arrays.

        Returns
        -------
        ndarray
            Smoothed P1D in Mpc, with shape ``(nk,)`` for one mapping or
            ``(nrow, nk)`` for a list.
        """

        type_data = type(data)
        if type_data is not list:
            data = [data]

        log_data = np.zeros((len(data), self.interp_logk_Mpc.shape[0]))
        for isim in range(len(data)):
            _ = data[isim]["k_Mpc"] > 0
            log_data[isim] = np.interp(
                self.interp_logk_Mpc,
                np.log(data[isim]["k_Mpc"][_]),
                np.log(data[isim]["p1d_Mpc"][_]),
            )

        # apply smoothing
        dat = skfda.FDataGrid(log_data, grid_points=self.interp_logk_Mpc)
        for ii in range(len(self.krange) - 1):
            _ = (self.interp_logk_Mpc > np.log(self.krange[ii])) & (
                self.interp_logk_Mpc <= np.log(self.krange[ii + 1])
            )
            log_data[:, _] = (
                self.kernel[ii].transform(dat).data_matrix[:, :, 0][:, _]
            )

        logk_Mpc = np.log(k_Mpc)
        data_smooth = np.zeros((len(data), k_Mpc.shape[0]))
        for isim in range(len(data)):
            data_smooth[isim] = np.exp(
                np.interp(logk_Mpc, self.interp_logk_Mpc, log_data[isim])
            )

        if type_data is not list:
            data_smooth = data_smooth[0]

        return data_smooth
