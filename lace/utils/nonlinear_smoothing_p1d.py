import numpy as np


class Nonlinear_Smoothing(object):
    """Smooth P1D using a nonlinear approach"""

    def __init__(
        self,
        data_set_kernel,
        kmax_Mpc,
        bandwidth=[0.8, 0.4, 0.2],
        krange=[0.15, 1, 2.5, 4],
    ):
        """Setup object either by passing measured power, or coefficients"""

        self.bandwidth = bandwidth
        self.krange = krange
        self.kmax_Mpc = kmax_Mpc

        log_data = self._interp_for_smoothing(data_set_kernel)
        self._set_kernel_smoothing(log_data)

    def _interp_for_smoothing(self, data):
        """Interpolate in logspace P1D to smooth it"""

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
        """Set kernel for smoothing"""
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
        """Apply kernel smoothing to data"""

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
