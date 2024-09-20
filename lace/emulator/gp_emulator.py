import GPy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import json
import time
from warnings import warn
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d

from lace.archive import gadget_archive
from lace.emulator import base_emulator
from lace.utils import poly_p1d
from lace.utils.nonlinear_smoothing_p1d import Nonlinear_Smoothing


class GPEmulator(base_emulator.BaseEmulator):
    """
    Initialize the Gaussian Process emulator.

    :param archive: Data archive used for training the emulator. Required if not using training_set.
    :type archive: class, optional
    :param training_set: Specific training set. Options are 'Perdersen21' and 'Cabayol23'.
    :type training_set: str, optional
    :param emulator_label: Specific emulator label. Options are 'Pedersen21', 'Pedersen23', 'Cabayol23'.
    :type emulator_label: str, optional
    :param emu_type: Type of emulator. Defaults to 'polyfit'.
    :type emu_type: str, optional
    :param verbose: Whether to print verbose messages. Defaults to False.
    :type verbose: bool, optional
    :param kmax_Mpc: Maximum k in Mpc^-1 for training. Defaults to 10.0.
    :type kmax_Mpc: float, optional
    :param emu_params: List of emulator parameters.
    :type emu_params: list, optional
    :param drop_sim: Specific simulations to drop.
    :type drop_sim: optional
    :param set_noise_var: Noise variance for emulator. Defaults to 1e-3.
    :type set_noise_var: float, optional
    :param check_hull: Whether to check if emulator calls are within the convex hull. Defaults to False.
    :type check_hull: bool, optional
    :param emu_per_k: Whether to build a GP for each k-bin. Defaults to False.
    :type emu_per_k: bool, optional
    :param ndeg: Degree of polynomial for fitting. Defaults to 4.
    :type ndeg: int, optional
    :param smoothing_bn: Bandwidths for smoothing.
    :type smoothing_bn: list, optional
    :param smoothing_krange: K-ranges for smoothing.
    :type smoothing_krange: list, optional
    """

    def __init__(
        self,
        archive=None,
        training_set=None,
        emulator_label=None,
        emu_type="polyfit",
        verbose=False,
        kmax_Mpc=10.0,
        emu_params=None,
        drop_sim=None,
        set_noise_var=1e-3,
        check_hull=False,
        emu_per_k=False,
        ndeg=4,
        smoothing_bn=None,
        smoothing_krange=None,
    ):
        self.kmax_Mpc = kmax_Mpc
        self.emu_type = emu_type
        self.emu_noise = set_noise_var
        self.verbose = verbose
        self.emu_per_k = emu_per_k
        self.ndeg = ndeg
        self.drop_sim = drop_sim
        self.smoothing_bn = smoothing_bn
        self.smoothing_krange = smoothing_krange

        # check input #
        training_set_all = ["Pedersen21", "Cabayol23"]
        if (archive != None) & (training_set != None):
            raise ValueError(
                "Conflict! Both custom archive and training_set provided"
            )

        if training_set is not None:
            if training_set in training_set_all:
                print(f"Selected training set from {training_set}")
            else:
                raise ValueError(
                    "Invalid training_set value. Available options: ",
                    training_set_all,
                )

            # read Gadget archive with the right postprocessing
            archive = gadget_archive.GadgetArchive(postproc=training_set)

        elif (archive is not None) and (training_set is None):
            print("Use custom archive provided by the user")
            pass

        elif (archive is None) & (training_set is None):
            raise (ValueError("Archive or training_set must be provided"))

        self.list_sim_cube = archive.list_sim_cube
        self.kp_Mpc = archive.kp_Mpc

        emulator_label_all = [
            "Pedersen21",
            "Pedersen23",
            "Pedersen21_ext",
            "Pedersen23_ext",
            "Pedersen21_ext8",
            "Pedersen23_ext8",
            "CH24",
        ]
        if emulator_label is not None:
            if emulator_label in emulator_label_all:
                print(f"Select emulator in {emulator_label}")
            else:
                raise ValueError(
                    "Invalid emulator_label value. Available options: ",
                    emulator_label_all,
                )

            if emulator_label == "Pedersen21":
                print(
                    r"Gaussian Process emulator predicting the P1D at each k-bin."
                    + " It goes to scales of 3Mpc^{-1} and z<=4.5. The parameters "
                    + "passed to the emulator will be overwritten to match these ones."
                )
                self.emu_params = [
                    "Delta2_p",
                    "n_p",
                    "mF",
                    "sigT_Mpc",
                    "gamma",
                    "kF_Mpc",
                ]
                self.zmax, self.kmax_Mpc, self.emu_type = 4.5, 3, "k_bin"
            elif emulator_label == "Pedersen23":
                print(
                    r"Gaussian Process emulator predicting the P1D, "
                    + "fitting coefficients to a 4th degree polynomial. It "
                    + "goes to scales of 3Mpc^{-1} and z<=4.5. The parameters"
                    + " passed to the emulator will be overwritten to match "
                    + "these ones"
                )

                self.emu_params = [
                    "Delta2_p",
                    "n_p",
                    "mF",
                    "sigT_Mpc",
                    "gamma",
                    "kF_Mpc",
                ]
                self.zmax, self.kmax_Mpc, self.ndeg, self.emu_type = (
                    4.5,
                    3,
                    4,
                    "polyfit",
                )
            elif emulator_label == "Pedersen21_ext":
                print(
                    r"Gaussian Process emulator predicting the P1D at each k-bin."
                    + " It goes to scales of 4Mpc^{-1} and z<=4.5. The parameters "
                    + "passed to the emulator will be overwritten to match these ones."
                )
                self.emu_params = [
                    "Delta2_p",
                    "n_p",
                    "mF",
                    "sigT_Mpc",
                    "gamma",
                    "kF_Mpc",
                ]
                self.zmax, self.kmax_Mpc, self.emu_type = 4.5, 4, "k_bin"

            elif emulator_label == "Pedersen23_ext":
                print(
                    r"Gaussian Process emulator predicting the P1D, "
                    + "fitting coefficients to a 5th degree polynomial. It "
                    + "goes to scales of 4Mpc^{-1} and z<=4.5. The parameters"
                    + " passed to the emulator will be overwritten to match "
                    + "these ones"
                )

                self.emu_params = [
                    "Delta2_p",
                    "n_p",
                    "mF",
                    "sigT_Mpc",
                    "gamma",
                    "kF_Mpc",
                ]
                self.zmax, self.kmax_Mpc, self.ndeg, self.emu_type = (
                    4.5,
                    4,
                    5,
                    "polyfit",
                )

            elif emulator_label == "Pedersen21_ext8":
                print(
                    r"Gaussian Process emulator predicting the P1D at each k-bin."
                    + " It goes to scales of 8 Mpc^{-1} and z<=4.5. The parameters "
                    + "passed to the emulator will be overwritten to match these ones."
                )
                self.emu_params = [
                    "Delta2_p",
                    "n_p",
                    "mF",
                    "sigT_Mpc",
                    "gamma",
                    "kF_Mpc",
                ]
                self.zmax, self.kmax_Mpc, self.emu_type = 4.5, 8, "k_bin"

            elif emulator_label == "Pedersen23_ext8":
                print(
                    r"Gaussian Process emulator predicting the P1D, "
                    + "fitting coefficients to a 5th degree polynomial. It "
                    + "goes to scales of 8 Mpc^{-1} and z<=4.5. The parameters"
                    + " passed to the emulator will be overwritten to match "
                    + "these ones"
                )

                self.emu_params = [
                    "Delta2_p",
                    "n_p",
                    "mF",
                    "sigT_Mpc",
                    "gamma",
                    "kF_Mpc",
                ]
                self.zmax, self.kmax_Mpc, self.ndeg, self.emu_type = (
                    4.5,
                    8,
                    7,
                    "polyfit",
                )

            elif emulator_label == "CH24":
                print(
                    r"Gaussian Process emulator predicting the P1D at each k-bin after "
                    + " applying smoothing. It goes to scales of 4 Mpc^{-1} and z<=4.5."
                    + "The parameters passed to the emulator will be overwritten to match these ones."
                )
                self.emu_params = [
                    "Delta2_p",
                    "n_p",
                    "mF",
                    "sigT_Mpc",
                    "gamma",
                    "kF_Mpc",
                ]
                self.zmax, self.kmax_Mpc, self.emu_type = (4.5, 4, "k_bin_sm")
                # bandwidth for different k-ranges
                self.smoothing_bn = [0.8, 0.4, 0.2]
                self.smoothing_krange = [0.15, 1, 2.5, 4]

        else:
            print("Selected custom emulator")

        # If none, take all parameters
        if emu_params == None:
            self.emu_params = [
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
                "Delta2_p",
                "n_p",
            ]
        else:
            self.emu_params = emu_params

        # GPs should probably avoid rescalings (low performance with large N)
        average = "both"
        val_scaling = 1
        if archive.training_average != "both":
            warn("Enforce average=both in training of GP emulator")
        if archive.training_val_scaling != 1:
            warn("Enforce val_scalinge=1 in training of GP emulator")

        # keep track of training data to be used in emulator
        self.training_data = archive.get_training_data(
            emu_params=self.emu_params,
            drop_sim=self.drop_sim,
            average=average,
            val_scaling=val_scaling,
        )
        self.kmin_Mpc = self.training_data[0]["k_Mpc"][1]

        ## Find max k bin
        self.k_bin = (
            np.max(np.where(self.training_data[0]["k_Mpc"] < self.kmax_Mpc)) + 1
        )
        self.training_k_bins = self.training_data[0]["k_Mpc"][1 : self.k_bin]

        self._build_interp()

        # train emulator
        self.train()

        # to check if emulator calls are in convex hull
        if check_hull:
            self.hull = Delaunay(self.X_param_grid)
        else:
            self.hull = None

    def _training_points_k_bin(self):
        """
        Get the training points for k-bin emulation in the form of P1D values at different k values.

        :return: Array of P1D values for each training data set.
        :rtype: numpy.ndarray
        """
        P1D_k = np.empty([len(self.training_data), self.k_bin - 1])
        for aa in range(len(self.training_data)):
            P1D_k[aa] = self.training_data[aa]["p1d_Mpc"][1 : self.k_bin]

        return P1D_k

    def _training_points_polyfit(self):
        """
        Get the training points for polynomial fitting in the form of polynomial coefficients.

        :return: Array of polynomial coefficients for each training data set.
        :rtype: numpy.ndarray
        """
        self._fit_p1d_in_archive(self.ndeg, self.kmax_Mpc)
        coeffs = np.empty([len(self.training_data), self.ndeg + 1])
        for aa in range(len(self.training_data)):
            coeffs[aa] = self.training_data[aa][
                "fit_p1d"
            ]  ## Collect P1D data for all k bins

        return coeffs

    def _training_points_k_bin_sm(self):
        """
        Get the training points for k-bin smoothing in the form of smoothed P1D values.

        :return: Array of smoothed P1D values for each training data set.
        :rtype: numpy.ndarray
        """
        self._k_bin_sm_p1d_in_archive(self.kmax_Mpc)
        coeffs = np.empty([len(self.training_data), self.k_bin - 1])
        for aa in range(len(self.training_data)):
            coeffs[aa] = self.training_data[aa]["p1d_sm"]

        return coeffs

    def _rescale_params(self, params):
        """
        Rescale a set of parameters to a unit volume.

        :param params: Parameters to be rescaled.
        :type params: list or numpy.ndarray
        :return: Rescaled parameters.
        :rtype: numpy.ndarray
        """
        for aa in range(len(params)):
            params[aa] = (params[aa] - self.param_limits[aa, 0]) / (
                self.param_limits[aa, 1] - self.param_limits[aa, 0]
            )

        return params

    def _buildTrainingSets(self):
        """
        Build training sets containing the parameter grid and corresponding training points.

        :return: Tuple containing:
            - Parameter grid for training.
            - Training points.
        :rtype: tuple
            - numpy.ndarray: Parameter grid for training.
            - numpy.ndarray: Training points.
        """
        ## Grid that will contain all training params
        params = np.empty([len(self.training_data), len(self.emu_params)])

        if self.emu_type == "k_bin":
            trainingPoints = self._training_points_k_bin()
        elif self.emu_type == "polyfit":
            trainingPoints = self._training_points_polyfit()
        elif self.emu_type == "k_bin_sm":
            trainingPoints = self._training_points_k_bin_sm()
        else:
            print("Unknown emulator type, terminating")
            quit()

        for aa in range(len(self.training_data)):
            for bb in range(len(self.emu_params)):
                params[aa][bb] = self.training_data[aa][
                    self.emu_params[bb]
                ]  ## Populate parameter grid

        return params, trainingPoints

    def _fit_p1d_in_archive(self, deg, kmax_Mpc):
        """
        Fit a polynomial to the logarithm of P1D for each entry in the archive.

        This method fits a polynomial to the logarithm of the power spectrum data (`p1d_Mpc`)
        for each entry in the `training_data` using a polynomial of the specified degree (`deg`),
        over the range of k values up to `kmax_Mpc`.

        :param deg: Degree of the polynomial for fitting.
        :type deg: int
        :param kmax_Mpc: Maximum k value in Mpc^-1 for fitting.
        :type kmax_Mpc: float
        """
        for entry in self.training_data:
            k_Mpc = entry["k_Mpc"]
            p1d_Mpc = entry["p1d_Mpc"]
            fit_p1d = poly_p1d.PolyP1D(
                k_Mpc, p1d_Mpc, kmin_Mpc=1.0e-3, kmax_Mpc=kmax_Mpc, deg=deg
            )
            entry[
                "fit_p1d"
            ] = fit_p1d.lnP_fit  ## Add coeffs for each model to archive

    def _k_bin_sm_p1d_in_archive(self, kmax_Mpc):
        """
        Apply smoothing to P1D data for each entry in the archive.

        This method performs kernel smoothing on the P1D data for each entry in the `training_data`
        using the specified maximum k value (`kmax_Mpc`), and updates each entry with the smoothed data.

        :param kmax_Mpc: Maximum k value in Mpc^-1 for smoothing.
        :type kmax_Mpc: float
        """
        # smoothing
        self.Kernel_Smoothing = Nonlinear_Smoothing(
            data_set_kernel=self.training_data,
            kmax_Mpc=self.kmax_Mpc,
            bandwidth=self.smoothing_bn,
            krange=self.smoothing_krange,
        )

        data_smooth = self.Kernel_Smoothing.apply_kernel_smoothing(
            self.training_k_bins, self.training_data
        )

        for isim, entry in enumerate(self.training_data):
            entry["p1d_sm"] = data_smooth[isim]

    def _build_interp(self):
        """
        Build Gaussian Process (GP) models from training data and parameter grid.

        This method constructs Gaussian Process models based on the training data and parameter grid.
        It involves rescaling parameters, normalizing training data, and initializing the GP models.
        Depending on the `emu_per_k` flag, it either builds a separate GP model for each k-bin
        or a single GP model for all k-bins.

        :return: None
        """
        self.X_param_grid, self.Ypoints = self._buildTrainingSets()

        ## Get parameter limits for rescaling
        self.param_limits = self._get_param_limits(self.X_param_grid)

        ## Rescaling to unit volume
        for cc in range(len(self.training_data)):
            self.X_param_grid[cc] = self._rescale_params(self.X_param_grid[cc])
        if self.verbose:
            print("Rescaled params to unity volume")

        ## Factors by which to rescale the flux to set a mean of 0
        self.scalefactors = np.median(self.Ypoints, axis=0)

        # Normalise by the median value
        self.normspectra = (self.Ypoints / self.scalefactors) - 1.0

        kernel = GPy.kern.RBF(len(self.emu_params), ARD=True)

        if self.emu_per_k:
            ## Build a GP for each k bin
            self.gp = []
            for aa in range(len(self.training_k_bins)):
                p1d_k = self.normspectra[:, aa]
                self.gp.append(
                    GPy.models.GPRegression(
                        self.X_param_grid,
                        p1d_k[:, None],
                        kernel=kernel,
                        noise_var=self.emu_noise,
                        initialize=False,
                    )
                )
        else:
            self.gp = GPy.models.GPRegression(
                self.X_param_grid,
                self.normspectra,
                kernel=kernel,
                noise_var=self.emu_noise,
                initialize=False,
            )

        return

    def _get_param_limits(self, paramGrid):
        """
        Get the minimum and maximum values for each parameter.

        This method computes the minimum and maximum values for each parameter in the provided
        parameter grid (`paramGrid`), which is used for rescaling parameters to a unit volume.

        :param paramGrid: 2D array where each column represents a parameter and each row represents a training point.
        :type paramGrid: numpy.ndarray
        :return: An array where each row contains the minimum and maximum values for a parameter.
        :rtype: numpy.ndarray
        """
        param_limits = np.empty((np.shape(paramGrid)[1], 2))
        for aa in range(len(param_limits)):
            param_limits[aa, 0] = min(paramGrid[:, aa])
            param_limits[aa, 1] = max(paramGrid[:, aa])

        return param_limits

    def train(self):
        """
        Train the Gaussian Process (GP) emulator.

        This method initializes and optimizes the Gaussian Process models. If `emu_per_k` is True,
        it trains multiple GP models, one for each k-bin. Otherwise, it trains a single GP model
        for all k-bins.

        :return: None
        """
        if self.emu_per_k:
            start = time.time()
            for gp in self.gp:
                gp.initialize_parameter()
                print("Training GP on %d points" % len(self.training_data))
                status = gp.optimize(messages=False)
                print("Optimised")
            end = time.time()
            print("all GPs optimised in {0:.2f} seconds".format(end - start))
        else:
            start = time.time()
            self.gp.initialize_parameter()
            print("Training GP on %d points" % len(self.training_data))
            status = self.gp.optimize(messages=False)
            end = time.time()
            print("GPs optimised in {0:.2f} seconds".format(end - start))

        return

    def printPriorVolume(self):
        """
        Print the limits for each parameter.

        This method prints the minimum and maximum values for all parameters used in the emulator.
        This provides a way to inspect the parameter space covered by the emulator.

        :return: None
        """
        for aa in range(len(self.emu_params)):
            print(self.emu_params[aa], self.param_limits[aa])

    def return_unit_call(self, model):
        """
        For a given model in dictionary format, return an ordered parameter list with the values rescaled to unit volume.

        This method takes a model dictionary, rescales the parameter values to a unit volume,
        and returns the list of rescaled parameter values.

        :param model: Dictionary containing parameter values with keys as parameter names.
        :type model: dict
        :return: List of rescaled parameter values.
        :rtype: list
        """
        param = []
        for aa, par in enumerate(self.emu_params):
            ## Rescale input parameters
            param.append(model[par])
            param[aa] = (param[aa] - self.param_limits[aa, 0]) / (
                self.param_limits[aa, 1] - self.param_limits[aa, 0]
            )
        return param

    def check_in_hull(self, model):
        """
        Check if a given model is within the convex hull of the training points.

        This method checks if the rescaled parameter values of a given model are within the
        convex hull of the training points' parameter space.

        :param model: Dictionary containing parameter values with keys as parameter names.
        :type model: dict
        :return: True if the model is inside the convex hull, False otherwise.
        :rtype: bool
        """
        param = self.return_unit_call(model)
        outside_hull = (
            self.hull.find_simplex(np.array(param).reshape(1, -1)) < 0
        )
        return not outside_hull

    def predict(self, model, z=None):
        """
        Return P1D or polynomial fit coefficients for a given parameter set.

        This method provides predictions of P1D values or polynomial coefficients based on
        the provided model parameters. It can also return error estimates if required.

        :param model: Dictionary containing parameter values with keys as parameter names.
        :type model: dict
        :param z: Optional parameter for rescaling, not fully tested.
        :type z: optional
        :return: Tuple containing:
            - Predicted P1D values.
            - Error estimates for the predictions.
        :rtype: tuple
            - numpy.ndarray: Predicted P1D values.
            - numpy.ndarray: Error estimates for the predictions.
        """

        try:
            length = len(model[self.emu_params[0]])
        except:
            length = 1

        emu_call = np.zeros((length, len(self.emu_params)))
        for ii, param in enumerate(self.emu_params):
            emu_call[:, ii] = model[param]

        emu_call = (emu_call - self.param_limits[:, 0]) / (
            self.param_limits[:, 1] - self.param_limits[:, 0]
        )

        # emu_per_k and reduce_var_* options only valid for k_bin emulator
        if self.emu_per_k:
            pred = np.zeros((length, len(self.gp)))
            var = np.array((length, len(self.gp)))
            for ii, gp in enumerate(self.gp):
                pred_single, var_single = gp.predict(emu_call)
                pred[:, ii] = pred_single
                var[:, ii] = var_single
        else:
            pred, var = self.gp.predict(emu_call)

        out_pred = (pred + 1) * self.scalefactors
        out_err = np.sqrt(var) * self.scalefactors

        return out_pred, out_err

    def emulate_p1d_Mpc(self, model, k_Mpc, return_covar=False, z=None):
        """
        Return the trained P(k) for an arbitrary set of k bins by interpolating the trained data.

        Optionally compute covariance if `return_covar` is True.

        :param model: Dictionary containing parameter values with keys as parameter names.
        :type model: dict
        :param k_Mpc: Array of k values in Mpc^-1 for which to predict P(k).
        :type k_Mpc: numpy.ndarray
        :param return_covar: Whether to return the covariance matrix. Defaults to False.
        :type return_covar: bool, optional
        :param z: Optional parameter for rescaling, not fully tested.
        :type z: optional
        :return: Tuple containing:
            - Predicted P1D values.
            - Covariance matrix if `return_covar` is True.
        :rtype: tuple
            - numpy.ndarray: Predicted P1D values.
            - numpy.ndarray (optional): Covariance matrix if `return_covar` is True.
        """

        for param in self.emu_params:
            if param not in model:
                raise ValueError(f"{param} not in input model")

        if np.max(k_Mpc) > self.kmax_Mpc:
            warn(
                f"Some of the requested k's are higher than the maximum training value k={self.kmax_Mpc}",
            )
        elif np.min(k_Mpc) < self.kmin_Mpc:
            warn(
                f"Some of the requested k's are lower than the minimum training value k={self.kmin_Mpc}"
            )

        try:
            length = len(model[self.emu_params[0]])
        except:
            length = 1

        if k_Mpc.ndim == 1:
            k_Mpc = np.repeat(k_Mpc[None, :], length, axis=0)

        # only implemented for length == 1
        if self.hull:
            if length == 1:
                # check if outside the convex hull
                if not self.check_in_hull(model):
                    print(z, "outside hull", model)

        # get raw prediction from GPy object
        gp_pred, gp_err = self.predict(model, z)

        if (self.emu_type == "k_bin") | (self.emu_type == "k_bin_sm"):
            # interpolate predictions to input k values
            interpolator = interp1d(
                self.training_k_bins,
                gp_pred,
                kind="cubic",
                fill_value="extrapolate",
            )
            p1d = np.zeros_like(k_Mpc)
            for ii in range(k_Mpc.shape[0]):
                p1d[ii] = interpolator(k_Mpc[ii])

            if length == 1:
                p1d = p1d[0]

            if not return_covar:
                return p1d
            else:
                # compute emulator covariance
                err_interp = interp1d(
                    self.training_k_bins,
                    gp_err,
                    kind="cubic",
                    fill_value="extrapolate",
                )
                p1d_err = np.zeros_like(k_Mpc)
                for ii in range(k_Mpc.shape[0]):
                    p1d_err[ii] = err_interp(k_Mpc[ii])

                if self.emu_per_k:
                    covar = np.diag(p1d_err**2)
                else:
                    # assume fully correlated errors when using same hyperparams
                    covar = np.zeros(
                        (k_Mpc.shape[0], k_Mpc.shape[1], k_Mpc.shape[1])
                    )
                    for ii in range(k_Mpc.shape[0]):
                        covar[ii] = np.outer(p1d_err[ii], p1d_err[ii])

                    if length == 1:
                        covar = covar[0]

                return p1d, covar

        elif self.emu_type == "polyfit":
            # gp_pred here are just the coefficients of the polynomial
            p1d = np.zeros((gp_pred.shape[0], k_Mpc.shape[1]))
            for ii in range(gp_pred.shape[0]):
                poly = np.poly1d(gp_pred[ii])
                p1d[ii] = np.exp(poly(np.log(k_Mpc[ii])))

            if not return_covar:
                if length == 1:
                    p1d = p1d[0]
                return p1d

            covar = np.zeros((gp_pred.shape[0], k_Mpc.shape[1], k_Mpc.shape[1]))
            for ii in range(gp_pred.shape[0]):
                lk = np.log(k_Mpc[ii])
                erry2 = (
                    (gp_err[ii, 0] * lk**4) ** 2
                    + (gp_err[ii, 1] * lk**3) ** 2
                    + (gp_err[ii, 2] * lk**2) ** 2
                    + (gp_err[ii, 3] * lk) ** 2
                    + gp_err[ii, 4] ** 2
                )
                # compute error on P
                err = p1d[ii] * np.sqrt(erry2)
                covar[ii] = np.outer(err, err)

            if length == 1:
                p1d = p1d[0]
                covar = covar[0]

            return p1d, covar

        else:
            raise ValueError("wrong emulator type")

    def get_nearest_distance(self, model, z=None):
        """
        Get the Euclidean distance to the nearest training point in the rescaled parameter space.

        This method computes the Euclidean distance from the given model parameters to the nearest
        training point in the rescaled parameter space.

        :param model: Dictionary containing parameter values with keys as parameter names.
        :type model: dict
        :param z: Optional parameter for rescaling, not fully tested.
        :type z: optional
        :return: Euclidean distance to the nearest training point.
        :rtype: float
        """

        param = []  ## List of input emulator parameter values
        ## First rescale the input model to unit volume
        for aa, par in enumerate(self.emu_params):
            ## Rescale input parameters
            param.append(model[par])
            param[aa] = (param[aa] - self.param_limits[aa, 0]) / (
                self.param_limits[aa, 1] - self.param_limits[aa, 0]
            )

        ## Find the closest training point, and find the Euclidean
        ## distance to that point
        shortest_distance = 99.99  ## Initialise variable
        for training_point in self.X_param_grid:
            ## Get Euclidean distance between the training point and
            ## the prediction point
            new_distance = np.sqrt(np.sum((training_point - param) ** 2))
            if new_distance < shortest_distance:
                shortest_distance = new_distance

        return shortest_distance

    def get_param_dict(self, point_number):
        """
        Return a dictionary with the emulator parameters for a given training point.

        This method returns a dictionary containing the parameter values for a specific training point.

        :param point_number: Index of the training point.
        :type point_number: int
        :return: Dictionary with parameter names as keys and their corresponding values.
        :rtype: dict
        """

        model_dict = {}
        for param in self.emu_params:
            model_dict[param] = self.training_data[point_number][param]

        return model_dict
