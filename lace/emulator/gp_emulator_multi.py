import pickle, os, time
from pathlib import Path
import numpy as np
from warnings import warn
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from lace.configuration import get_data_path
from lace.emulator.model_manifest import ModelBundleError, load_manifest, write_manifest


def func_poly(x, a, b, c, d, e):
    """Evaluate the five-component sigmoid basis used for P1D fitting.

    Parameters
    ----------
    x : array-like
        Dimensionless wavenumber ``k / kmax``.
    a, b, c, d, e : float or array-like
        Coefficients of sigmoid components with fixed exponents.

    Returns
    -------
    ndarray
        Fitted logarithmic P1D shape with NumPy broadcasting.
    """
    return (
        a / (1 + np.exp(0.5 * x))
        + b / (1 + np.exp(1 * x))
        + c / (1 + np.exp(2 * x))
        + d / (1 + np.exp(4 * x))
        + e / (1 + np.exp(8 * x))
    )


def optimizer(obj_func, x0, bounds):
    """Minimize a differentiable objective with L-BFGS-B.

    Parameters
    ----------
    obj_func : callable
        Objective returning ``(value, gradient)``.
    x0 : array-like
        Initial parameter vector.
    bounds : sequence
        Bounds accepted by :func:`scipy.optimize.minimize`.

    Returns
    -------
    x, fun : tuple
        Optimized coordinates and final scalar objective value.
    """
    res = minimize(
        obj_func,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000},
    )
    return res.x, res.fun


class GPEmulator:
    """Gaussian-process emulator for normalized one-dimensional flux power.

    The emulator predicts P1D in Mpc units from the suite-specific linear,
    thermal, and mean-flux parameter vectors selected by ``emulator_label``.
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
        model_path=None,
        data_path=None,
        normalization_path=None,
    ):
        """Load a production bundle or train a GP emulator from archive entries.

        Parameters
        ----------
        archive, archive2 : BaseArchive, optional
            Training archive(s); ``archive2`` is used only by the combined
            ``CH24_gpr`` model.
        emulator_label : str, default="CH24_nyx_gpr"
            Supported GP model family.
        drop_sim : str, optional
            Simulation omitted for a leave-one-out model.
        train, save : bool, default=False
            Train a model instead of loading it; ``save`` is retained for API
            compatibility while trained models are written by this class.
        n_restarts_optimizer : int, default=0
            Number of GP kernel optimizer restarts.
        model_path, data_path, normalization_path : path-like, optional
            Explicit model root, data root, and normalization-file locations.

        Raises
        ------
        ValueError
            If model labels or training inputs are invalid.
        FileNotFoundError
            If a required production bundle or normalization asset is absent.
        """
        self.emulator_label = emulator_label
        self.drop_sim = drop_sim
        self.n_restarts_optimizer = n_restarts_optimizer
        self.archive = archive
        self.archive2 = archive2

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

        self.data_path = get_data_path(data_path)
        if model_path is None:
            folder_save = self.data_path / "GPmodels" / emulator_label
        else:
            if isinstance(model_path, str) and not model_path.strip():
                raise ValueError("model_path must not be blank")
            folder_save = Path(model_path).expanduser()
        if train:
            folder_save.mkdir(parents=True, exist_ok=True)
            print("Storing emulator in " + str(folder_save))
        elif not folder_save.is_dir():
            raise FileNotFoundError(
                f"Model directory does not exist: {folder_save}. Pass model_path, "
                "configure paths.data_path, or install a trusted external model bundle."
            )
        if drop_sim is None:
            label = "full.pkl"
            label_meta = "meta.npy"
        else:
            label = "drop_" + self.drop_sim + ".pkl"
            label_meta = "meta_" + self.drop_sim + ".npy"

        self.folder_save = str(folder_save)
        self.label = label
        self.path_save_meta = os.path.join(folder_save, label_meta)
        self.normalization_path = self._normalization_path(normalization_path, folder_save)
        if not train:
            self.manifest = load_manifest(
                self.folder_save, self.emulator_label, self.label, self.normalization_path
            )
            if self.manifest is None:
                warn(
                    f"Loading legacy model bundle at {self.folder_save} without a manifest. "
                    "Only use model files from a trusted source; provenance and compatibility are unknown.",
                    UserWarning,
                )

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
            self.input_norm = self._load_normalization(self.normalization_path)
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
            self.input_norm = self._load_normalization(self.normalization_path)
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
            self.input_norm = self._load_normalization(self.normalization_path)
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

        self._prediction_cache = None

    def _normalization_path(self, normalization_path, folder_save):
        """Resolve the normalization asset for this model bundle.

        Explicit paths take precedence; otherwise a self-contained external
        model root is preferred before the configured LaCE data directory.
        """
        if normalization_path is not None:
            if isinstance(normalization_path, str) and not normalization_path.strip():
                raise ValueError("normalization_path must not be blank")
            return Path(normalization_path).expanduser()
        # A self-contained external model root may keep ff_mpgcen.npy beside GPmodels.
        inferred = Path(folder_save).parent.parent / "ff_mpgcen.npy"
        return inferred if inferred.is_file() else self.data_path / "ff_mpgcen.npy"

    @staticmethod
    def _load_normalization(path):
        """Load and validate the trusted normalization dictionary from disk.

        Parameters
        ----------
        path : path-like
            NumPy serialization containing normalization arrays.

        Returns
        -------
        dict
            Stored normalization data.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Normalization file is missing: {path}. Provide normalization_path or "
                "configure a data_path containing ff_mpgcen.npy."
            )
        try:
            return np.load(path, allow_pickle=True).item()
        except PermissionError as error:
            raise PermissionError(f"Cannot read normalization file {path}") from error
        except (OSError, ValueError, EOFError, pickle.UnpicklingError) as error:
            raise ModelBundleError(f"Corrupt normalization file {path}: {error}") from error

    def _save_emu(self):
        """Serialize trained GP experts, metadata, and a provenance manifest."""
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
        files = [Path(self.path_save_meta)] + [
            Path(self.folder_save) / ("n" + str(ii) + "_" + self.label)
            for ii in range(len(self.gp))
        ]
        provenance = {
            "archive_class": type(self.archive).__name__,
            "postproc": getattr(self.archive, "postproc", None),
            "nyx_version": getattr(self.archive, "nyx_version", None),
            "kp_Mpc": self.kp_Mpc,
            "average": self.average,
            "val_scaling": self.val_scaling,
            "z_max": self.z_max,
        }
        write_manifest(
            self.folder_save, self.emulator_label, self.label, files,
            self.normalization_path, self.drop_sim, provenance,
        )

    def _load_emu(self):
        """Load checked metadata and serialized GP experts for this bundle."""
        # The manifest was validated before normalization and model deserialization.
        path_meta = Path(self.path_save_meta)
        if not path_meta.is_file():
            raise FileNotFoundError(f"Model metadata is missing: {path_meta}. Restore a complete trusted model bundle.")
        try:
            metadata = np.load(path_meta, allow_pickle=True).item()
        except PermissionError as error:
            raise PermissionError(f"Cannot read model metadata {path_meta}") from error
        except (OSError, ValueError, EOFError, pickle.UnpicklingError) as error:
            raise ModelBundleError(f"Corrupt model metadata {path_meta}: {error}") from error
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
            try:
                with open(path_save_gp, "rb") as f:
                    self.gp.append(pickle.load(f))
            except FileNotFoundError as error:
                raise FileNotFoundError(f"Model checkpoint is missing: {path_save_gp}. Restore a complete trusted bundle.") from error
            except PermissionError as error:
                raise PermissionError(f"Cannot read model checkpoint {path_save_gp}") from error
            except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError) as error:
                raise ModelBundleError(f"Cannot load model checkpoint {path_save_gp}: {error}") from error

    def _training_points_gpolyfit(self, training_data):
        """Fit every archive row and return its GP training targets.

        Parameters
        ----------
        training_data : sequence of mapping
            Archive entries containing P1D, k grid, and emulator inputs.

        Returns
        -------
        ndarray
            Polynomial coefficients or sampled normalized shapes, one row per
            input archive entry.
        """

        store_fit = self._gfit_p1d_in_archive(training_data)

        return store_fit

    def _buildTrainingSets(self, training_data):
        """Build finite emulator-input and target arrays from archive entries.

        Parameters
        ----------
        training_data : sequence of mapping
            Archive rows from which emulator parameters and P1D targets are
            extracted.

        Returns
        -------
        params, training_points : tuple of ndarray
            Finite physical input matrix of shape ``(nrow, nparam)`` and
            matching target matrix.
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
        """Fit normalized P1D shapes for all selected archive entries.

        Parameters
        ----------
        training_data : sequence of mapping
            Entries containing ``k_Mpc``, ``p1d_Mpc``, and mean flux.

        Returns
        -------
        ndarray
            Five sigmoid coefficients for ``gpolyfit`` or normalized P1D values
            on ``k_Mpc_emu`` for ``gkbin``.
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
        """Standardize training data and fit overlapping mean-flux GP experts.

        Parameters
        ----------
        training_data : sequence of mapping
            Selected finite archive entries used to train the emulator.

        Returns
        -------
        None
            Sets standardized training arrays, expert-bin boundaries, and
            fitted :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
            instances.
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

    def _prepare_model_input(self, model):
        """Validate scalar or columnar emulator inputs and stack their rows.

        Parameters
        ----------
        model : mapping
            Every emulator parameter as a scalar or matching one-dimensional
            column.

        Returns
        -------
        ndarray, int
            Physical input matrix with shape ``(nrow, nparam)`` and ``nrow``.
        """
        if not isinstance(model, dict):
            raise TypeError("model must be a dictionary of emulator parameters")
        values = []
        length = None
        for param in self.emu_params:
            if param not in model:
                raise ValueError(f"{param} not in input model")
            value = np.asarray(model[param])
            if value.ndim == 0:
                value = value.reshape(1)
            elif value.ndim != 1:
                raise ValueError(f"{param} must be a scalar or one-dimensional array, not shape {value.shape}")
            if length is None:
                length = value.size
            elif value.size != length:
                raise ValueError(f"All emulator parameter arrays must have the same length; {param} has {value.size}, expected {length}")
            values.append(value)
        try:
            return np.column_stack(values).astype(float), length
        except (TypeError, ValueError) as error:
            raise ValueError("Emulator parameters must be numeric scalars or one-dimensional arrays") from error

    def _prediction_key(self, values):
        """Return a hashable cache key for one physical emulator-input row."""

        return tuple(float(value) for value in values)

    def _predict_rows(self, physical_rows):
        """Predict coefficient rows with overlapping mean-flux GP experts.

        Parameters
        ----------
        physical_rows : ndarray
            Input matrix with shape ``(nrow, nparam)`` in physical units.

        Returns
        -------
        ndarray
            Unstandardized polynomial coefficients or k-bin targets with one
            row per input.
        """

        emu_call = (physical_rows - self.xscalings_mean) / self.xscalings_std
        length = emu_call.shape[0]
        scaled_prediction = np.zeros((length, len(self.tscalings_mean)))
        counts = np.zeros(length)
        mean_flux = emu_call[:, self.ind_mF]
        central_expert = np.argmin(
            np.abs(mean_flux[:, None] - self.bin_mF_cen[None, :]), axis=1
        )

        rows_by_expert = [[] for _ in self.gp]
        for row, expert in enumerate(central_expert):
            rows_by_expert[expert].append(row)
            if expert != 0 and mean_flux[row] < self.bin_mF_top[expert - 1]:
                rows_by_expert[expert - 1].append(row)
            if (
                expert != len(self.bin_mF_cen) - 1
                and mean_flux[row] > self.bin_mF_bot[expert + 1]
            ):
                rows_by_expert[expert + 1].append(row)

        for expert, rows in enumerate(rows_by_expert):
            if not rows:
                continue
            rows = np.asarray(rows, dtype=int)
            scaled_prediction[rows] += self.gp[expert].predict(emu_call[rows])
            counts[rows] += 1

        return (
            scaled_prediction / counts[:, None]
        ) * self.tscalings_std + self.tscalings_mean

    def prime_prediction_cache(self, emulator_calls):
        """Precompute unique emulator rows needed by a batched caller.

        Parameters
        ----------
        emulator_calls : iterable of mapping
            Scalar or columnar emulator input mappings.

        Returns
        -------
        None
            Stores predictions keyed by their physical input rows.
        """

        unique_rows = {}
        for model in emulator_calls:
            rows, _ = self._prepare_model_input(model)
            for row in rows:
                unique_rows.setdefault(self._prediction_key(row), row)
        if not unique_rows:
            self._prediction_cache = {}
            return
        keys = list(unique_rows)
        predictions = self._predict_rows(np.asarray([unique_rows[key] for key in keys]))
        self._prediction_cache = {
            key: predictions[index] for index, key in enumerate(keys)
        }

    def clear_prediction_cache(self):
        """Discard transient predictions retained for a batched likelihood call."""

        self._prediction_cache = None

    def predict(self, model):
        """Return GP target predictions for scalar or columnar emulator inputs.

        Parameters
        ----------
        model : mapping
            Scalar or one-dimensional columns for every emulator parameter.

        Returns
        -------
        ndarray
            Coefficient or k-bin target matrix with shape ``(nrow, ntarget)``.
        """

        physical_rows, _ = self._prepare_model_input(model)
        if self._prediction_cache is None:
            return self._predict_rows(physical_rows)
        return np.asarray(
            [
                self._prediction_cache[self._prediction_key(row)]
                for row in physical_rows
            ]
        )

    def emulate_p1d_Mpc(self, model, k_Mpc, verbose=False, return_coeff=False):
        """Evaluate emulated one-dimensional flux power in comoving units.

        Parameters
        ----------
        model : mapping
            Scalar or columnar emulator parameter inputs.
        k_Mpc : array-like
            One shared ``(nk,)`` grid or per-row ``(nrow, nk)`` grids in 1/Mpc.
        verbose : bool, default=False
            Warn when requested wavenumbers exceed the training interval.
        return_coeff : bool, default=False
            Also return GP polynomial or k-bin coefficients.

        Returns
        -------
        ndarray or tuple of ndarray
            P1D in Mpc, shape ``(nrow, nk)``, optionally followed by the GP
            coefficient matrix.
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

        _, length = self._prepare_model_input(model)
        k_Mpc = np.asarray(k_Mpc)
        if k_Mpc.ndim == 1:
            k_Mpc = np.repeat(k_Mpc[None, :], length, axis=0)
        elif k_Mpc.ndim != 2 or k_Mpc.shape[0] != length:
            raise ValueError(f"k_Mpc must be one-dimensional or have {length} rows; got shape {k_Mpc.shape}")

        # get raw prediction from GP object
        gp_pred = self.predict(model)

        p1d = np.zeros((gp_pred.shape[0], k_Mpc.shape[1]))

        for ii in range(gp_pred.shape[0]):
            mF_values = np.asarray(model["mF"])
            mF = mF_values.item() if mF_values.ndim == 0 else mF_values[ii]

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

    def emulate_P1D_Mpc(
        self,
        model,
        k_iMpc,
        verbose=False,
        return_coeff=False,
        return_covar=False,
        z=None,
    ):
        """Evaluate P1D at ``k_iMpc`` using the canonical unit-bearing API.

        ``return_covar`` and ``z`` belonged to a retired generic interface.
        This GP emulator does not implement either feature, so accepting them
        would silently change the meaning of other arguments.

        Parameters
        ----------
        model, k_iMpc, verbose, return_coeff
            Forwarded to :meth:`emulate_p1d_Mpc`; ``k_iMpc`` is in 1/Mpc.
        return_covar : bool, default=False
            Unsupported covariance request.
        z : float, optional
            Unsupported redshift override.

        Returns
        -------
        ndarray or tuple of ndarray
            Emulated P1D in Mpc, optionally with GP coefficients.
        """

        if return_covar:
            raise NotImplementedError(
                "GPEmulator does not provide P1D covariance predictions"
            )
        if z is not None:
            raise NotImplementedError(
                "GPEmulator does not support redshift overrides in P1D predictions"
            )
        return self.emulate_p1d_Mpc(
            model, k_iMpc, verbose=verbose, return_coeff=return_coeff
        )
