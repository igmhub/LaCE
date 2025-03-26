import GPy, pickle, os, time
import numpy as np
from warnings import warn
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import lace
from lace.emulator import base_emulator


def func_poly1(x, a, b, c, d, e, f):
    return a + b * x**0.5 + c * x + d * x**2 + e * x**3 + f * x**4


# def func_poly2(x, a, b, c, d, e, f, g):
#     return (
#         a
#         + b * x**0.5
#         + c * x**0.75
#         + d * x
#         + e * x**2
#         + f * x**3
#         + g * x**4
#     )


def func_poly2(x, a, b, c, d, e, f, g):
    return (
        a
        + b * x**0.5
        + c * x**0.75
        + d * x
        + e * x**2
        + f * x**3
        + g * x**4
    )


class GPEmulator(base_emulator.BaseEmulator):
    """
    Initialize the Gaussian Process emulator.
    """

    def __init__(
        self,
        archive=None,
        emulator_label="CH24_mpg_gp",
        drop_sim=None,
        train=False,
        save=False,
    ):
        self.emulator_label = emulator_label
        self.drop_sim = drop_sim

        # check emulator
        emulator_label_all = ["CH24_mpg_gp", "CH24_nyx_gp"]
        if self.emulator_label in emulator_label_all:
            print(f"Select emulator in {self.emulator_label}")
        else:
            raise ValueError(
                "Invalid emulator_label value. Available options: ",
                emulator_label_all,
            )

        repo = os.path.dirname(lace.__path__[0])
        folder_save = os.path.join(repo, "data", "GPmodels", emulator_label)
        os.makedirs(folder_save, exist_ok=True)
        if train:
            print("Storing emulator in " + folder_save)
        self.path_save_meta = os.path.join(folder_save, "meta.npy")
        if drop_sim is None:
            label = "full.pkl"
        else:
            label = self.drop_sim + ".pkl"
        self.path_save_gp = os.path.join(folder_save, label)

        if self.emulator_label == "CH24_mpg_gp":
            self.emu_params = [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]
            self.zmax, self.kmax_Mpc, self.emu_type = (5.5, 5, "gpolyfit")
            self.average = "both"
            self.val_scaling = None
            # smoothing function
            self.func_poly = func_poly1
            self.ndeg = 6
            # normalization
            fname = os.path.join(repo, "data", "ff_mpgcen.npy")
            self.input_norm = np.load(fname, allow_pickle=True).item()
            self.norm_imF = interp1d(
                self.input_norm["mF"], self.input_norm["p1d_Mpc_mF"], axis=0
            )

        elif self.emulator_label == "CH24_nyx_gp":
            self.emu_params = [
                "Delta2_p",
                "n_p",
                "alpha_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]

            self.zmax, self.kmax_Mpc, self.emu_type = (5.5, 5, "gpolyfit")
            self.average = "both"
            self.val_scaling = None
            # smoothing function
            self.func_poly = func_poly1
            self.ndeg = 6
            # normalization
            fname = os.path.join(repo, "data", "ff_mpgcen.npy")
            self.input_norm = np.load(fname, allow_pickle=True).item()
            self.norm_imF = interp1d(
                self.input_norm["mF"], self.input_norm["p1d_Mpc_mF"], axis=0
            )

        if train == False:
            self._load_emu()

        else:
            if archive is None:
                raise ValueError("Archive must be provided for training")

            # XXX save as metadata
            self.list_sim_cube = archive.list_sim_cube
            self.kp_Mpc = archive.kp_Mpc

            # keep track of training data to be used in emulator
            training_data = archive.get_training_data(
                emu_params=self.emu_params,
                drop_sim=self.drop_sim,
                average=self.average,
                val_scaling=self.val_scaling,
            )

            k_Mpc = training_data[0]["k_Mpc"]
            # do we need this condition?
            self.kmin_Mpc = k_Mpc[k_Mpc > 0].min()

            self._initialize(training_data)
            self._train()
            self._save_emu()

    def _save_emu(self):
        # save emulator
        with open(self.path_save_gp, "wb") as f:
            pickle.dump(self.gp, f)

        # save metadata
        if self.drop_sim is None:
            metadata = {}
            metadata["kmin_Mpc"] = self.kmin_Mpc
            metadata["param_limits"] = self.param_limits
            metadata["tscalings_std"] = self.tscalings_std
            metadata["tscalings_mean"] = self.tscalings_mean
            metadata["kp_Mpc"] = self.kp_Mpc
            metadata["list_sim_cube"] = self.list_sim_cube
            np.save(self.path_save_meta, metadata)

    def _load_emu(self):
        # load emulator
        self.gp = pickle.load(open(self.path_save_gp, "rb"))

        # load metadata
        metadata = np.load(self.path_save_meta, allow_pickle=True).item()
        self.kmin_Mpc = metadata["kmin_Mpc"]
        self.param_limits = metadata["param_limits"]
        self.tscalings_std = metadata["tscalings_std"]
        self.tscalings_mean = metadata["tscalings_mean"]
        self.kp_Mpc = metadata["kp_Mpc"]
        self.list_sim_cube = metadata["list_sim_cube"]

    def _training_points_gpolyfit(self, training_data):
        """
        Get the training points for polynomial fitting in the form of polynomial coefficients.

        :return: Array of polynomial coefficients for each training data set.
        :rtype: numpy.ndarray
        """

        store_fit = self._gfit_p1d_in_archive(training_data)
        self.tscalings_mean = np.mean(store_fit, axis=0)
        self.tscalings_std = np.std(store_fit, axis=0)

        return store_fit

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

    def _buildTrainingSets(self, training_data):
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
        params = np.empty([len(training_data), len(self.emu_params)])

        trainingPoints = self._training_points_gpolyfit(training_data)

        for aa in range(len(training_data)):
            for bb in range(len(self.emu_params)):
                params[aa][bb] = training_data[aa][
                    self.emu_params[bb]
                ]  ## Populate parameter grid

        return params, trainingPoints

    def _gfit_p1d_in_archive(self, training_data):
        """
        Fit a function to the logarithm of P1D for each entry in the archive.
        """

        store_fit = np.zeros((len(training_data), self.ndeg))

        for ii, entry in enumerate(training_data):
            ind_k = (entry["k_Mpc"] > 0) & (entry["k_Mpc"] < self.kmax_Mpc)
            # ind_k = entry["k_Mpc"] < self.kmax_Mpc
            k_Mpc = entry["k_Mpc"][ind_k]
            k_fit = k_Mpc / self.kmax_Mpc

            norm = np.interp(
                k_Mpc, self.input_norm["k_Mpc"], self.norm_imF(entry["mF"])
            )

            y2fit = np.log(entry["p1d_Mpc"][ind_k] / norm)
            store_fit[ii], _ = curve_fit(self.func_poly, k_fit, y2fit)

        return store_fit

    def _initialize(self, training_data):
        """
        Build Gaussian Process (GP) models from training data and parameter grid.

        This method constructs Gaussian Process models based on the training data and parameter grid.
        It involves rescaling parameters, normalizing training data, and initializing the GP models.
        Depending on the `emu_per_k` flag, it either builds a separate GP model for each k-bin
        or a single GP model for all k-bins.

        :return: None
        """
        self.X_param_grid, self.Ypoints = self._buildTrainingSets(training_data)

        ## Get parameter limits for rescaling
        self.param_limits = self._get_param_limits(self.X_param_grid)

        ## Rescaling data
        for cc in range(len(training_data)):
            self.X_param_grid[cc] = self._rescale_params(self.X_param_grid[cc])

        self.normspectra = (
            self.Ypoints - self.tscalings_mean
        ) / self.tscalings_std

        # kernel = GPy.kern.RBF(len(self.emu_params), ARD=True)
        kernel = GPy.kern.Matern32(len(self.emu_params), ARD=True)
        # kernel = GPy.kern.Matern52(len(self.emu_params), ARD=True)

        self.gp = GPy.models.GPRegression(
            self.X_param_grid,
            self.normspectra,
            kernel=kernel,
            initialize=False,
        )

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

    def _train(self):
        """
        Train the Gaussian Process (GP) emulator.

        This method initializes and optimizes the Gaussian Process models. If `emu_per_k` is True,
        it trains multiple GP models, one for each k-bin. Otherwise, it trains a single GP model
        for all k-bins.

        :return: None
        """

        start = time.time()
        self.gp.initialize_parameter()
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

    def predict(self, model):
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
        pred, var = self.gp.predict(emu_call)

        out_pred = pred * self.tscalings_std + self.tscalings_mean
        out_err = np.sqrt(var) * self.tscalings_std

        return out_pred, out_err

    def emulate_p1d_Mpc(self, model, k_Mpc):
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

        # get raw prediction from GPy object
        gp_pred, gp_err = self.predict(model)

        p1d = np.zeros((gp_pred.shape[0], k_Mpc.shape[1]))

        for ii in range(gp_pred.shape[0]):
            p1d_unorm = self.func_poly(k_Mpc[ii] / self.kmax_Mpc, *gp_pred[ii])

            try:
                mF = model["mF"][ii]
            except:
                mF = model["mF"]

            norm = np.interp(
                k_Mpc[ii], self.input_norm["k_Mpc"], self.norm_imF(mF)
            )
            p1d[ii] = np.exp(p1d_unorm) * norm

        return p1d
