import GPy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import json
import time
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
from lace.emulator import p1d_archive
from lace.emulator import poly_p1d
from lace.archive import pnd_archive


class GPEmulator:
    """
    Gaussian process emulator to emulate P1D from a simulation suite.
    This will train on the data in an 'archive' object, and will return
    a given P_1D(k) for the same k-bins used in training.
    GPEmulator.predict takes models in a dictionary format currently.
    """

    def __init__(
        self,
        basedir="/lace/emulator/sim_suites/Australia20/",
        p1d_label=None,
        skewers_label=None,
        verbose=False,
        kmax_Mpc=10.0,
        paramList=None,
        train=True,
        drop_tau_rescalings=True,
        drop_temp_rescalings=True,
        emu_type="polyfit",
        z_max=5,
        archive=None,
        set_noise_var=1e-3,
        asymmetric_kernel=True,
        check_hull=False,
        set_hyperparams=None,
        paramLimits=None,
        rbf_only=True,
        emu_per_k=False,
        ndeg=4,
        postprocessing="768",
        drop_sim=None,
    ):
        self.kmax_Mpc = kmax_Mpc
        self.basedir = basedir
        self.emu_type = emu_type
        self.emu_noise = set_noise_var
        self.drop_tau_rescalings = drop_tau_rescalings
        self.drop_temp_rescalings = drop_temp_rescalings
        self.verbose = verbose
        self.asymmetric_kernel = asymmetric_kernel
        self.z_max = z_max
        self.paramLimits = paramLimits
        self.postprocessing = postprocessing
        self.crossval = False  ## Flag to check whether or not a prediction is
        ## inside the training set
        self.rbf_only = rbf_only
        self.emu_per_k = emu_per_k

        if self.postprocessing == "768":
            self.key_data = "data_av_all"  # Selects average of the three axes and two phases in the new postprocessing

        elif self.postprocessing == "500":
            self.key_data = "data"  ##Select data of the old postprocessing
        self.ndeg = ndeg

        # read all files with P1D measured in simulation suite
        if archive == None:
            self.custom_archive = False
            if self.postprocessing == "500":
                self.archive = pnd_archive.archivePND(sim_suite="Pedersen21")
            elif self.postprocessing == "768":
                archive = pnd_archive.archivePND(
                    z_max=4.5, nsamples=30, drop_sim=drop_sim
                )

                archive.average_over_samples(flag="all")
                archive.average_over_samples(flag="phases")
                archive.average_over_samples(flag="axes")
                archive.input_emulator(flag="all")
                archive.input_emulator(flag="phases")
                archive.input_emulator(flag="axes")

                archive.training_data = [d for d in archive.data_av_all]
                if drop_tau_rescalings == True:
                    archive.training_data = [
                        d for d in archive.training_data if d["scale_tau"] == 1
                    ]
                self.archive = archive

            else:
                raise ('Available archives are "500" and "768"')

        else:
            self.custom_archive = True
            if self.verbose:
                print(
                    "Loading emulator using a specific archive, not the one set in basedir"
                )
            self.archive = archive

        ## Find max k bin
        self.k_bin = (
            np.max(np.where(self.archive.training_data[0]["k_Mpc"] < self.kmax_Mpc)) + 1
        )
        self.training_k_bins = self.archive.training_data[0]["k_Mpc"][1 : self.k_bin]

        ## If none, take all parameters
        if paramList == None:
            self.paramList = ["mF", "sigT_Mpc", "gamma", "kF_Mpc", "Delta2_p", "n_p"]
        else:
            self.paramList = paramList

        self._build_interp(self.archive, self.paramList)
        self.trained = False

        if train == True:
            self.train()

        # to check if emulator calls are in convex hull
        if check_hull:
            self.hull = Delaunay(self.X_param_grid)
        else:
            self.hull = None

        self.emulators = None  ## Flag that this is an individual emulator object

    def _training_points_k_bin(self, archive):
        """Method to get the Y training points in the form of the P1D
        at different k values"""

        P1D_k = np.empty([len(self.archive.training_data), self.k_bin - 1])
        for aa in range(len(self.archive.training_data)):
            P1D_k[aa] = self.archive.training_data[aa]["p1d_Mpc"][1 : self.k_bin]

        return P1D_k

    def _training_points_polyfit(self, archive):
        """Method to get the Y training points in the form of polyfit
        coefficients"""

        self._fit_p1d_in_archive(self.ndeg, self.kmax_Mpc)
        coeffs = np.empty([len(self.archive.training_data), self.ndeg + 1])
        for aa in range(len(self.archive.training_data)):
            coeffs[aa] = self.archive.training_data[aa][
                "fit_p1d"
            ]  ## Collect P1D data for all k bins

        return coeffs

    def _rescale_params(self, params, paramLimits):
        """Rescale a set of parameters to have a unit volume"""

        for aa in range(len(params)):
            params[aa] = (params[aa] - paramLimits[aa, 0]) / (
                paramLimits[aa, 1] - paramLimits[aa, 0]
            )

        return params

    def _buildTrainingSets(self, archive, paramList):
        """Build the grids that contain the training parameters
        This is a nxm grid of X data (n for number of training points, m
        for number of parameters), and a length nxk set of Y  data, k being
        the number of k bins for the k bin emulator, or number of polynomial
        coefficients for the polyfit emulator"""

        ## Grid that will contain all training params
        params = np.empty([len(self.archive.training_data), len(paramList)])

        if self.emu_type == "k_bin":
            trainingPoints = self._training_points_k_bin(archive)
        elif self.emu_type == "polyfit":
            trainingPoints = self._training_points_polyfit(archive)
        else:
            print("Unknown emulator type, terminating")
            quit()

        for aa in range(len(self.archive.training_data)):
            for bb in range(len(paramList)):
                params[aa][bb] = self.archive.training_data[aa][
                    paramList[bb]
                ]  ## Populate parameter grid

        return params, trainingPoints

    def _fit_p1d_in_archive(self, deg, kmax_Mpc):
        """For each entry in archive, fit polynomial to log(p1d)"""

        for entry in self.archive.training_data:
            k_Mpc = entry["k_Mpc"]
            p1d_Mpc = entry["p1d_Mpc"]
            fit_p1d = poly_p1d.PolyP1D(
                k_Mpc, p1d_Mpc, kmin_Mpc=1.0e-3, kmax_Mpc=kmax_Mpc, deg=deg
            )
            entry["fit_p1d"] = fit_p1d.lnP_fit  ## Add coeffs for each model to archive

    def _build_interp(self, archive, paramList):
        """Method to build an GP object from a spectra archive and list of parameters
        Currently the parameter rescaling is done by taking the min and max
        of the provided params, not by defining our own prior volume. Need to decide
        whether or not this is what we want."""

        self.X_param_grid, self.Ypoints = self._buildTrainingSets(archive, paramList)

        ## Get parameter limits for rescaling
        if self.paramLimits is None:
            self.paramLimits = self._get_param_limits(self.X_param_grid)

        ## Rescaling to unit volume
        for cc in range(len(self.archive.training_data)):
            self.X_param_grid[cc] = self._rescale_params(
                self.X_param_grid[cc], self.paramLimits
            )
        if self.verbose:
            print("Rescaled params to unity volume")

        ## Factors by which to rescale the flux to set a mean of 0
        self.scalefactors = np.median(self.Ypoints, axis=0)

        # Normalise by the median value
        self.normspectra = (self.Ypoints / self.scalefactors) - 1.0

        if self.rbf_only == False:
            kernel = GPy.kern.Linear(len(paramList), ARD=self.asymmetric_kernel)
            kernel += GPy.kern.RBF(len(paramList), ARD=self.asymmetric_kernel)
        else:
            kernel = GPy.kern.RBF(len(paramList), ARD=self.asymmetric_kernel)

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
        """Get the min and max values for each parameter"""

        paramLimits = np.empty((np.shape(paramGrid)[1], 2))
        for aa in range(len(paramLimits)):
            paramLimits[aa, 0] = min(paramGrid[:, aa])
            paramLimits[aa, 1] = max(paramGrid[:, aa])

        return paramLimits

    def train(self):
        """Train the GP emulator"""

        if self.emu_per_k:
            start = time.time()
            for gp in self.gp:
                gp.initialize_parameter()
                print("Training GP on %d points" % len(self.archive.training_data))
                status = gp.optimize(messages=False)
                print("Optimised")
            end = time.time()
            print("all GPs optimised in {0:.2f} seconds".format(end - start))
        else:
            start = time.time()
            self.gp.initialize_parameter()
            print("Training GP on %d points" % len(self.archive.training_data))
            status = self.gp.optimize(messages=False)
            end = time.time()
            print("GPs optimised in {0:.2f} seconds".format(end - start))

        self.trained = True

        return

    def printPriorVolume(self):
        """Print the limits for each parameter"""

        for aa in range(len(self.paramList)):
            print(self.paramList[aa], self.paramLimits[aa])

    def return_unit_call(self, model):
        """For a given model in dictionary format, return an
        ordered parameter list with the values rescaled to unit volume
        """

        param = []
        for aa, par in enumerate(self.paramList):
            ## Rescale input parameters
            param.append(model[par])
            param[aa] = (param[aa] - self.paramLimits[aa, 0]) / (
                self.paramLimits[aa, 1] - self.paramLimits[aa, 0]
            )
        return param

    def check_in_hull(self, model):
        param = self.return_unit_call(model)
        outside_hull = self.hull.find_simplex(np.array(param).reshape(1, -1)) < 0
        return not outside_hull

    def predict(self, model, z=None):
        """Return P1D or polyfit coeffs for a given parameter set
        For the k bin emulator this will be in the training k bins
        Option to pass 'z' for rescaling is not fully tested."""

        if self.trained == False:
            print("Emulator not trained, cannot make a prediction")
            return
        param = []
        for aa, par in enumerate(self.paramList):
            ## Rescale input parameters
            param.append(model[par])
            param[aa] = (param[aa] - self.paramLimits[aa, 0]) / (
                self.paramLimits[aa, 1] - self.paramLimits[aa, 0]
            )
        ## Check if model is inside training set
        if self.crossval == True:
            isin = np.isin(param, self.X_param_grid)
            if np.sum(isin) == len(param):  ## Check all parameters
                print("Emulator call is inside training set!!!")

        # emu_per_k and reduce_var_* options only valid for k_bin emulator
        if self.emu_per_k:
            pred = np.array([])
            var = np.array([])
            for gp in self.gp:
                pred_single, var_single = gp.predict(np.array(param).reshape(1, -1))
                pred = np.append(pred, pred_single)
                var = np.append(var, var_single)
        else:
            pred, var = self.gp.predict(np.array(param).reshape(1, -1))

        out_pred = np.ndarray.flatten((pred + 1) * self.scalefactors)
        out_err = np.ndarray.flatten(np.sqrt(var) * self.scalefactors)

        return out_pred, out_err

    def emulate_p1d_Mpc(self, model, k_Mpc, return_covar=False, z=None, old_cov=False):
        """
        Method to return the trained P(k) for an arbitrary set of k bins
        by interpolating the trained data.
        Option for reducing variance with z rescaling is not fully tested.
        Kept option old_cov=True only to reproduce old (bad) results.
        """
        try:
            if max(k_Mpc) > max(self.training_k_bins):
                print(max(k_Mpc))
                print(max(self.training_k_bins))
                print(
                    "Warning! Your requested k bins are higher than the training values."
                )
        except:
            if k_Mpc > max(self.training_k_bins):
                print(max(k_Mpc))
                print(max(self.training_k_bins))
                print(
                    "Warning! Your requested k bins are higher than the training values."
                )

        if self.hull:
            # check if outside the convex hull
            if not self.check_in_hull(model):
                print(z, "outside hull", model)

        # get raw prediction from GPy object
        gp_pred, gp_err = self.predict(model, z)

        if self.emu_type == "k_bin":
            # interpolate predictions to input k values
            interpolator = interp1d(
                self.training_k_bins, gp_pred, kind="cubic", fill_value="extrapolate"
            )
            p1d = interpolator(k_Mpc)
            if not return_covar:
                return p1d
            # compute emulator covariance
            err_interp = interp1d(
                self.training_k_bins, gp_err, kind="cubic", fill_value="extrapolate"
            )
            p1d_err = err_interp(k_Mpc)
            if self.emu_per_k:
                covar = np.diag(p1d_err**2)
            else:
                # assume fully correlated errors when using same hyperparams
                covar = np.outer(p1d_err, p1d_err)
            return p1d, covar

        elif self.emu_type == "polyfit":
            # gp_pred here are just the coefficients of the polynomial
            poly = np.poly1d(gp_pred)
            p1d = np.exp(poly(np.log(k_Mpc)))
            if not return_covar:
                return p1d
            if old_cov:
                # old covariance (should not be used)
                err = np.abs(gp_err)
                err = (
                    err[0] * p1d**4
                    + err[1] * p1d**3
                    + err[2] * p1d**2
                    + err[3] * p1d
                )
                covar = np.outer(err, err)
            else:
                # first estimate error on y=log P
                lk = np.log(k_Mpc)
                erry2 = (
                    (gp_err[0] * lk**4) ** 2
                    + (gp_err[1] * lk**3) ** 2
                    + (gp_err[2] * lk**2) ** 2
                    + (gp_err[3] * lk) ** 2
                    + gp_err[4] ** 2
                )
                # compute error on P
                err = p1d * np.sqrt(erry2)
                covar = np.outer(err, err)
            return p1d, covar

        else:
            raise ValueError("wrong emulator type")

    def get_nearest_distance(self, model, z=None):
        """For a given model, get the Euclidean distance to the nearest
        training point (in the rescaled parameter space)"""

        param = []  ## List of input emulator parameter values
        ## First rescale the input model to unit volume
        for aa, par in enumerate(self.paramList):
            ## Rescale input parameters
            param.append(model[par])
            param[aa] = (param[aa] - self.paramLimits[aa, 0]) / (
                self.paramLimits[aa, 1] - self.paramLimits[aa, 0]
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
        """Return a dictionary with the emulator parameters
        for a given training point"""

        model_dict = {}
        for param in self.paramList:
            model_dict[param] = self.archive.training_data[point_number][param]

        return model_dict

    def saveEmulator(self):
        """Method to save a trained emulator. The emulator
        hyperparameters are saved alongside a .json dictionary
        detailing the initial configuration of the emulator.
        When an emulator is saved it will check the basedir
        for existing emulator saves.

        We currently do not save emulators on custom archives
        as it is impossible to know if the training points
        are all the same. Training points are currently
        reassembled using the snapshot archive, and not
        saved alongside the emulator hyperparameters."""

        ## Perform checks
        if self.custom_archive or self.max_archive_size:
            print("Cannot save emulators trained on custom archives")
            return
        if not self.trained:
            print("Cannot save an emulator that is not trained")
            return

        ## Create and save dictionary of emulator parameters
        ## Can't think of a way to do this iteratively so we write it out
        initParams = {}
        initParams["k_bin"] = self.k_bin
        initParams["emu_type"] = self.emu_type
        initParams["emu_noise"] = self.emu_noise
        initParams["drop_tau_rescalings"] = self.drop_tau_rescalings
        initParams["drop_temp_rescalings"] = self.drop_temp_rescalings
        initParams["keep_every_other_rescaling"] = self.keep_every_other_rescaling
        initParams["undersample_z"] = self.undersample_z
        initParams["paramList"] = self.paramList
        initParams["asymmetric_kernel"] = self.asymmetric_kernel
        initParams["z_max"] = self.z_max

        saveString = self.basedir + "/saved_emulator_"

        ## Here we check to see if an emulator matching
        ## our initial parameters is already saved
        aa = 1
        while os.path.isfile(saveString + str(aa) + ".json"):
            ## Load dictionary and check if the values are different
            ## to the emulator we want to save
            with open(saveString + str(aa) + ".json") as json_file:
                fileInitDict = json.load(json_file)
            if fileInitDict == initParams:  ## If so we can break the loop
                if self.verbose:
                    print("This emulator is already saved.")
                return
            else:  ## If not keep looking through saved emulators
                aa += 1

        saveString = saveString + str(aa)
        with open("%s.json" % saveString, "w") as fp:
            json.dump(initParams, fp)

        ## Create and save dictionary of hyperparameters
        saveDict = {}
        saveDict["paramList"] = self.paramList
        saveDict["kmax_Mpc"] = self.kmax_Mpc
        np.save("%s.npy" % saveString, self.gp.param_array)
        if self.verbose:
            print("Model saved as %s.npy" % saveString)

    def loadEmulator(self):
        """Method to load a saved set of emulator
        hyperparameters. We need to make sure the
        dataset and emulator configurations that the
        hyperparameters were optimised on is the same
        as the dataset we will use as training data.
        So we will check the .json files for a perfect match"""

        ## Perform same checks as when saving an emulator
        ## as save/load does not work with non-standard
        ## data archives

        if self.custom_archive or self.max_archive_size:
            print("Cannot load emulators with non-standard training data")
            return
        if self.trained:
            print("Cannot load an emulator after training")
            return

        initParams = {}
        initParams["k_bin"] = self.k_bin
        initParams["emu_type"] = self.emu_type
        initParams["emu_noise"] = self.emu_noise
        initParams["drop_tau_rescalings"] = self.drop_tau_rescalings
        initParams["drop_temp_rescalings"] = self.drop_temp_rescalings
        initParams["keep_every_other_rescaling"] = self.keep_every_other_rescaling
        initParams["undersample_z"] = self.undersample_z
        initParams["paramList"] = self.paramList
        initParams["asymmetric_kernel"] = self.asymmetric_kernel
        initParams["z_max"] = self.z_max

        saveString = self.basedir + "/saved_emulator_"

        ## Here we check to see if an emulator matching
        ## our initial parameters is already saved
        aa = 1
        while os.path.isfile(saveString + str(aa) + ".json"):
            ## Load dictionary and check if the values are different
            ## to the emulator we want to load
            with open(saveString + str(aa) + ".json") as json_file:
                fileInitDict = json.load(json_file)
            if fileInitDict == initParams:  ## If so, load it
                saveString = saveString + str(aa)
                self.gp.update_model(False)
                self.gp.initialize_parameter()
                self.gp[:] = np.load(saveString + ".npy")
                self.gp.update_model(True)
                self.trained = True
                if self.verbose:
                    print("Loading emulator from %s.npy" % saveString)
                return

            else:  ## If not keep looking through saved emulators
                aa += 1
        if self.verbose:
            print("Could not find a matching emulator to load")

    def load_default(self):
        """Load the default set of hyperparams and parameter limits
        for the given sim suite. This is the set of hyperparams trained
        on the full suite of sims, and allows us to standardise our emulator
        when testing on different sims and with different training sets"""

        # the only stored emulator is for old settings (obsolete)
        if self.asymmetric_kernel or self.rbf_only:
            raise ValueError("can not load emulator")

        ## Load saved emulator dictionary
        repo = os.environ["LACE_REPO"] + "/"
        emulator_path = repo + self.archive.basedir + "/emulator.json"

        with open(emulator_path, "r") as fp:
            emu_load = json.load(fp)

        ## Have to use asarray as json won't save numpy arrays
        ## but gp uses numpy arrays
        ## Make sure paramList is the same as the one the hyperparameters
        ## were optimised on
        assert self.paramList == [
            "mF",
            "sigT_Mpc",
            "gamma",
            "kF_Mpc",
            "Delta2_p",
            "n_p",
        ]
        self.load_hyperparams(
            np.asarray(emu_load["hyperparams"]), np.asarray(emu_load["paramLimits"])
        )

        return

    def load_hyperparams(self, hyperparams, paramLimits=None):
        """Load a specific set of emulator hyperparameters.
        Also have option to load an associated set of parameter limits.
        Will rebuilt the X training grid new parameter limits are passed
        """

        ## If we give a new set of paramlimits
        ## also reconstruct the training data
        if paramLimits is not None:
            self.paramLimits = paramLimits
            self._build_interp(self.archive, self.paramList)

        self.gp.update_model(False)
        self.gp.initialize_parameter()
        self.gp[:] = hyperparams
        self.gp.update_model(True)
        self.trained = True
        if self.verbose:
            print("Emulator hyperparameters loaded")

        return
