import pickle, os, time
import numpy as np
from warnings import warn
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
import lace
from lace.emulator import base_emulator


def func_poly(x, a, b, c, d, e):
    return (
        a / (1 + np.exp(0.5 * x))
        + b / (1 + np.exp(1 * x))
        + c / (1 + np.exp(2 * x))
        + d / (1 + np.exp(4 * x))
        + e / (1 + np.exp(8 * x))
    )


def optimizer(obj_func, x0, bounds):
    res = minimize(
        obj_func,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000},
    )
    return res.x, res.fun


class GPEmulator(base_emulator.BaseEmulator):
    """
    Initialize the Gaussian Process emulator.
    """

    def __init__(
        self,
        archive=None,
        archive2=None,
        emulator_label="CH24_nyx_gpr",
        drop_sim=None,
        train=False,
        save=False,
        n_restarts_optimizer=0,
    ):
        self.emulator_label = emulator_label
        self.drop_sim = drop_sim
        self.n_restarts_optimizer = n_restarts_optimizer

        # check emulator
        emulator_label_all = [
            "CH24_mpg_gpr",
            "CH24_nyx_gpr",
            "CH24_mpgcen_gpr",
            "CH24_nyxcen_gpr",
            "CH24_gpr",
        ]
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
        if drop_sim is None:
            label = "full.pkl"
            label_meta = "meta.npy"
        else:
            label = "drop_" + self.drop_sim + ".pkl"
            label_meta = "meta_" + self.drop_sim + ".npy"

        self.folder_save = folder_save
        self.label = label
        self.path_save_meta = os.path.join(folder_save, label_meta)

        if (self.emulator_label == "CH24_mpg_gpr") | (
            self.emulator_label == "CH24_mpgcen_gpr"
        ):
            self.emu_params = [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]
            self.z_max, self.kmin_Mpc, self.kmax_Mpc, self.emu_type = (
                5.5,
                0.02,
                4.25,
                "gpolyfit",
            )
            # self.z_max, self.kmax_Mpc, self.emu_type = (5.5, 4.25, "gkbin")
            self.average = "both"
            self.kernel = Matern(
                nu=0.5, length_scale=np.ones(len(self.emu_params))
            )
            self.nelem_max = 1200
            # self.val_scaling = 1
            self.val_scaling = None
            # smoothing function
            self.func_poly = func_poly
            self.ndeg = 5
            # normalization
            fname = os.path.join(repo, "data", "ff_mpgcen.npy")
            self.input_norm = np.load(fname, allow_pickle=True).item()
            self.norm_imF = interp1d(
                self.input_norm["mF"], self.input_norm["p1d_Mpc_mF"], axis=0
            )
            if self.emulator_label == "CH24_mpg_gpr":
                self.add_central = False
            else:
                self.add_central = True

        elif (self.emulator_label == "CH24_nyx_gpr") | (
            self.emulator_label == "CH24_nyxcen_gpr"
        ):
            self.emu_params = [
                "Delta2_p",
                "n_p",
                "alpha_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]

            self.z_max, self.kmin_Mpc, self.kmax_Mpc, self.emu_type = (
                5.5,
                0.02,
                4.25,
                "gpolyfit",
            )
            self.average = "both"
            self.val_scaling = None
            # self.kernel = Matern(
            #     nu=0.5, length_scale=np.ones(len(self.emu_params))
            # ) + WhiteKernel(noise_level_bounds=(1e-15, 1e2))
            self.kernel = Matern(
                nu=0.5, length_scale=np.ones(len(self.emu_params))
            )
            # smoothing function
            self.nelem_max = 1200
            self.func_poly = func_poly
            self.ndeg = 5
            # normalization
            fname = os.path.join(repo, "data", "ff_mpgcen.npy")
            self.input_norm = np.load(fname, allow_pickle=True).item()
            self.norm_imF = interp1d(
                self.input_norm["mF"], self.input_norm["p1d_Mpc_mF"], axis=0
            )
            if self.emulator_label == "CH24_nyx_gpr":
                self.add_central = False
            else:
                self.add_central = True

        elif self.emulator_label == "CH24_gpr":
            self.emu_params = [
                "Delta2_p",
                "n_p",
                "alpha_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                # "kF_Mpc",
            ]

            # self.z_max, self.kmax_Mpc, self.emu_type = (5.5, 4.25, "gpolyfit")
            self.z_max, self.kmin_Mpc, self.kmax_Mpc, self.emu_type = (
                5.5,
                0.02,
                4.25,
                "gkbin",
            )
            self.average = "both"
            if "mpg" in archive.data[0]["sim_label"]:
                self.val_scaling = 1
            else:
                self.val_scaling = None

            if "mpg" in archive2.data[0]["sim_label"]:
                self.val_scaling2 = 1
            else:
                self.val_scaling2 = None

            self.kernel = Matern(
                nu=0.5, length_scale=np.ones(len(self.emu_params))
            )
            # smoothing function
            self.func_poly = func_poly
            self.ndeg = 5
            # normalization
            fname = os.path.join(repo, "data", "ff_mpgcen.npy")
            self.input_norm = np.load(fname, allow_pickle=True).item()
            self.norm_imF = interp1d(
                self.input_norm["mF"], self.input_norm["p1d_Mpc_mF"], axis=0
            )
            self.add_central = False

        self.ind_mF = np.argwhere(np.array(self.emu_params) == "mF")[0, 0]

        if train == False:
            self._load_emu()

        else:
            if archive is None:
                raise ValueError("Archive must be provided for training")

            self.list_sim_cube = archive.list_sim_cube
            self.kp_Mpc = archive.kp_Mpc

            training_data = archive.get_training_data(
                emu_params=self.emu_params,
                drop_sim=self.drop_sim,
                average=self.average,
                val_scaling=self.val_scaling,
                z_max=self.z_max,
            )

            if self.add_central:
                # never add for l10
                if self.drop_sim is not None:
                    pass
                else:
                    if "mpg" in self.emulator_label:
                        cen_lab = "mpg_central"
                    else:
                        cen_lab = "nyx_central"

                    training_data += archive.get_testing_data(cen_lab)

            if self.emulator_label == "CH24_gpr":
                self.list_sim_cube += archive2.list_sim_cube
                if self.kp_Mpc != archive2.kp_Mpc:
                    raise ("Must be same kp_Mpc for archive and archive2")
                training_data += archive2.get_training_data(
                    emu_params=self.emu_params,
                    drop_sim=self.drop_sim,
                    average=self.average,
                    val_scaling=self.val_scaling2,
                    z_max=self.z_max,
                )

            self._initialize(training_data)
            self._save_emu()

    def _save_emu(self):
        # save emulator
        for ii in range(len(self.gp)):
            path_save_gp = os.path.join(
                self.folder_save, "n" + str(ii) + "_" + self.label
            )
            with open(path_save_gp, "wb") as f:
                pickle.dump(self.gp[ii], f, protocol=5)

        # save metadata
        metadata = {}
        metadata["n_emus"] = len(self.gp)
        metadata["kmin_Mpc"] = self.kmin_Mpc
        metadata["xscalings_std"] = self.xscalings_std
        metadata["xscalings_mean"] = self.xscalings_mean
        metadata["tscalings_std"] = self.tscalings_std
        metadata["tscalings_mean"] = self.tscalings_mean
        metadata["kp_Mpc"] = self.kp_Mpc
        metadata["list_sim_cube"] = self.list_sim_cube
        metadata["bin_mF_cen"] = self.bin_mF_cen
        metadata["bin_mF_bot"] = self.bin_mF_bot
        metadata["bin_mF_top"] = self.bin_mF_top
        if self.emu_type == "gkbin":
            metadata["k_Mpc_emu"] = self.k_Mpc_emu
        np.save(self.path_save_meta, metadata)

    def _load_emu(self):
        # load metadata
        metadata = np.load(self.path_save_meta, allow_pickle=True).item()
        self.kmin_Mpc = metadata["kmin_Mpc"]
        self.xscalings_mean = metadata["xscalings_mean"]
        self.xscalings_std = metadata["xscalings_std"]
        self.tscalings_mean = metadata["tscalings_mean"]
        self.tscalings_std = metadata["tscalings_std"]
        self.kp_Mpc = metadata["kp_Mpc"]
        self.list_sim_cube = metadata["list_sim_cube"]
        self.bin_mF_cen = metadata["bin_mF_cen"]
        self.bin_mF_bot = metadata["bin_mF_bot"]
        self.bin_mF_top = metadata["bin_mF_top"]
        n_emus = metadata["n_emus"]
        if self.emu_type == "gkbin":
            self.k_Mpc_emu = metadata["k_Mpc_emu"]

        # load emulator
        self.gp = []
        for ii in range(n_emus):
            path_save_gp = os.path.join(
                self.folder_save, "n" + str(ii) + "_" + self.label
            )
            with open(path_save_gp, "rb") as f:
                self.gp.append(pickle.load(f))

    def _training_points_gpolyfit(self, training_data):
        """
        Get the training points for polynomial fitting in the form of polynomial coefficients.

        :return: Array of polynomial coefficients for each training data set.
        :rtype: numpy.ndarray
        """

        store_fit = self._gfit_p1d_in_archive(training_data)

        return store_fit

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
        params = np.zeros((len(training_data), len(self.emu_params)))

        ## Populate parameter grid
        for ii in range(len(training_data)):
            for jj in range(len(self.emu_params)):
                params[ii, jj] = training_data[ii][self.emu_params[jj]]

        trainingPoints = self._training_points_gpolyfit(training_data)

        # curate data
        mask1 = np.all(np.isfinite(params), axis=1)
        mask2 = np.all(np.isfinite(trainingPoints), axis=1)
        mask = (mask1 == True) & (mask2 == True)
        params = params[mask]
        trainingPoints = trainingPoints[mask]
        print(params.shape, trainingPoints.shape)

        return params, trainingPoints

    def _gfit_p1d_in_archive(self, training_data):
        """
        Fit a function to the logarithm of P1D for each entry in the archive.
        """

        if self.emu_type == "gpolyfit":
            store_fit = np.zeros((len(training_data), self.ndeg))
        elif self.emu_type == "gkbin":
            self.k_Mpc_emu = np.logspace(
                np.log10(self.kmin_Mpc), np.log10(self.kmax_Mpc), 50
            )
            kpoly = self.k_Mpc_emu / self.kmax_Mpc

            store_fit = np.zeros((len(training_data), self.k_Mpc_emu.shape[0]))

        for ii, entry in enumerate(training_data):
            ind_k = (entry["k_Mpc"] > 0) & (entry["k_Mpc"] < self.kmax_Mpc)
            k_Mpc = entry["k_Mpc"][ind_k]
            k_fit = k_Mpc / self.kmax_Mpc

            norm = np.interp(
                k_Mpc, self.input_norm["k_Mpc"], self.norm_imF(entry["mF"])
            )

            y2fit = np.log(entry["p1d_Mpc"][ind_k] / norm)
            par_fit, _ = curve_fit(self.func_poly, k_fit, y2fit)

            if self.emu_type == "gpolyfit":
                store_fit[ii] = par_fit
            elif self.emu_type == "gkbin":
                store_fit[ii] = self.func_poly(kpoly, *par_fit)

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

        self.xpoints, self.tpoints = self._buildTrainingSets(training_data)
        ## Standarize (from all data, could be different for each GP)

        # x variables
        self.xscalings_mean = np.mean(self.xpoints, axis=0)
        self.xscalings_std = np.std(self.xpoints, axis=0)
        self.xpoints = (self.xpoints - self.xscalings_mean) / self.xscalings_std

        # y variables
        self.tscalings_mean = np.mean(self.tpoints, axis=0)
        self.tscalings_std = np.std(self.tpoints, axis=0)
        self.tpoints = (self.tpoints - self.tscalings_mean) / self.tscalings_std

        nelem = self.xpoints.shape[0]
        nn_gps = int(np.ceil(nelem / self.nelem_max))

        # bin data into percentiles
        list_percen = np.linspace(0, 100, nn_gps)
        df_percen = list_percen[1] - list_percen[0]
        list_percen_bot = list_percen - 0.5 * df_percen
        list_percen_bot[list_percen_bot < 0] = 0
        list_percen_top = list_percen + 0.5 * df_percen
        list_percen_top[list_percen_top > 100] = 100

        # impose some overlap
        list_percen_bot[1:] -= df_percen * 0.1
        list_percen_top[:-1] += df_percen * 0.1

        self.bin_mF_bot = np.percentile(
            self.xpoints[:, self.ind_mF], list_percen_bot
        )
        self.bin_mF_top = np.percentile(
            self.xpoints[:, self.ind_mF], list_percen_top
        )
        self.bin_mF_cen = 0.5 * (self.bin_mF_bot + self.bin_mF_top)

        self.gp = []
        start = time.time()
        for ii in range(nn_gps):
            ## Rescaling data

            mask = (self.xpoints[:, self.ind_mF] >= self.bin_mF_bot[ii]) & (
                self.xpoints[:, self.ind_mF] <= self.bin_mF_top[ii]
            )
            print(ii, mask.sum(), self.bin_mF_bot[ii], self.bin_mF_top[ii])

            self.gp.append(
                GaussianProcessRegressor(
                    kernel=self.kernel,
                    optimizer=optimizer,
                    random_state=0,
                    n_restarts_optimizer=self.n_restarts_optimizer,
                ).fit(
                    self.xpoints[mask],
                    self.tpoints[mask],
                )
            )

        end = time.time()
        print("GPs optimised in {0:.2f} seconds".format(end - start))

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

        # input
        emu_call = np.zeros((length, len(self.emu_params)))
        for ii, param in enumerate(self.emu_params):
            emu_call[:, ii] = model[param]
        emu_call = (emu_call - self.xscalings_mean) / (self.xscalings_std)

        # output
        out_pred = np.zeros((length, len(self.tscalings_mean)))
        for ii in range(length):
            mF = emu_call[ii, self.ind_mF]
            jj = np.argmin(np.abs(mF - self.bin_mF_cen))
            # print(mF, self.bin_mF_cen, self.bin_mF_cen[jj])

            # average emulator predictions if mF in overlapping range
            num = 1.0
            pred = self.gp[jj].predict(np.atleast_2d(emu_call[ii]))[0]
            if jj != 0:
                if mF < self.bin_mF_top[jj - 1]:
                    num += 1.0
                    pred += self.gp[jj - 1].predict(
                        np.atleast_2d(emu_call[ii])
                    )[0]
            if jj != (len(self.bin_mF_cen) - 1):
                if mF > self.bin_mF_bot[jj + 1]:
                    num += 1.0
                    pred += self.gp[jj + 1].predict(
                        np.atleast_2d(emu_call[ii])
                    )[0]

            # print(ii, jj, num, mF, self.bin_mF_bot[jj], self.bin_mF_top[jj])

            out_pred[ii] = (
                pred / num
            ) * self.tscalings_std + self.tscalings_mean

        return out_pred

    def emulate_p1d_Mpc(self, model, k_Mpc, verbose=False, return_coeff=False):
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

        if verbose:
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

        # get raw prediction from GP object
        gp_pred = self.predict(model)

        p1d = np.zeros((gp_pred.shape[0], k_Mpc.shape[1]))

        for ii in range(gp_pred.shape[0]):
            try:
                mF = model["mF"][ii]
            except:
                mF = model["mF"]

            norm = np.interp(
                k_Mpc[ii], self.input_norm["k_Mpc"], self.norm_imF(mF)
            )

            if self.emu_type == "gpolyfit":
                p1d_unorm = self.func_poly(
                    k_Mpc[ii] / self.kmax_Mpc, *gp_pred[ii]
                )
            elif self.emu_type == "gkbin":
                p1d_unorm = np.interp(k_Mpc[ii], self.k_Mpc_emu, gp_pred[ii])

            p1d[ii] = np.exp(p1d_unorm) * norm

        if return_coeff:
            return p1d, gp_pred
        else:
            return p1d
