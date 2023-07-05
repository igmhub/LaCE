import GPy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import json
import time
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
from lace.archive import gadget_archive
from lace.emulator import poly_p1d
from lace.emulator import base_emulator


class GPEmulator(base_emulator.BaseEmulator):
    """
    Gaussian process emulator to emulate P1D from a simulation suite.
    This will train on the data in an 'archive' object, and will return
    a given P_1D(k) for the same k-bins used in training.
    GPEmulator.predict takes models in a dictionary format currently.
    
    

    Args:
        archive (class): Data archive used for training the emulator.
            Required when using a custom emulator.
        training_set: Specific training set.  Options are
            'Perdersen21', 'Cabayol23.
        emu_params (lsit): A list of emulator parameters.
        emulator_label (str): Specific emulator label. Options are
            'Pedersen21' and 'Pedersen23'.
        kmax_Mpc (float): The maximum k in Mpc^-1 to use for training. Default is 3.5.

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
        set_noise_var=1e-3,
        asymmetric_kernel=True,
        check_hull=False,
        paramLimits=None,
        rbf_only=True,
        emu_per_k=False,
        ndeg=4,
    ):
        self.kmax_Mpc = kmax_Mpc
        self.emu_type = emu_type
        self.emu_noise = set_noise_var
        self.verbose = verbose
        self.asymmetric_kernel = asymmetric_kernel
        self.paramLimits = paramLimits
        self.rbf_only = rbf_only
        self.emu_per_k = emu_per_k
        self.ndeg = ndeg

        # check input #
        training_set_all = ["Pedersen21", "Cabayol23"]
        if (archive!=None)&(training_set!=None):
            raise ValueError("Conflict! Both custom archive and training_set provided")
        
        if training_set is not None:
            try:
                if training_set in training_set_all:
                    print(f"Selected training set from {training_set}")
                    pass
                else:
                    print(
                        "Invalid training_set value. Available options: ",
                        training_set_all,
                    )
                    raise
            except:
                print("An error occurred while checking the training_set value.")
                raise
                
            # read Gadget archive with the right postprocessing
            self.archive=gadget_archive.GadgetArchive(postproc=training_set)
                
        elif archive!=None and training_set==None:
            print("Use custom archive provided by the user")
            self.archive = archive

        elif (archive==None)&(training_set==None):
            raise(ValueError('Archive or training_set must be provided'))

        emulator_label_all = ["Pedersen21", "Pedersen23"]
        if emulator_label is not None:  
            try:
                if emulator_label in emulator_label_all:
                    print(f"Select emulator in {emulator_label}")
                    pass
                else:
                    print(
                        "Invalid emulator_label value. Available options: ",
                        emulator_label_all,
                    )
                    raise

            except:
                print("An error occurred while checking the emulator_label value.")
                raise

            if emulator_label == "Pedersen21":
                
                print(
                    r"Gaussian Process emulator predicting the P1D at each k-bin."
                    + " It goes to scales of 3Mpc^{-1} and z<4.5. The parameters "
                    + "passed to the emulator will be overwritten to match these ones."
                )
                self.emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
                self.zmax, self.kmax_Mpc, self.emu_type =  4.5, 3, "k_bin"
                
            if emulator_label == "Pedersen23":
                
                print(
                    r"Gaussian Process emulator predicting the optimal P1D"
                    + "fitting coefficients to a 5th degree polynomial. It "
                    + "goes to scales of 4Mpc^{-1} and z<4.5. The parameters"
                    + " passed to the emulator will be overwritten to match "
                    + "these ones"
                )
            
                self.emuparams = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
                self.zmax, self.kmax_Mpc, self.ndeg, self.empu_type = 4.5, 4, 4, "polyfit"
                                
                
        else:
            print("Selected custom emulator")      
            
        # keep track of training data to be used in emulator
        self.training_data = self.archive.get_training_data()

        ## Find max k bin
        self.k_bin = (
            np.max(np.where(self.training_data[0]["k_Mpc"] < self.kmax_Mpc)) + 1
        )
        self.training_k_bins = self.training_data[0]["k_Mpc"][1 : self.k_bin]

        ## If none, take all parameters
        if emu_params == None:
            self.emu_params = ["mF", "sigT_Mpc", "gamma", "kF_Mpc", "Delta2_p", "n_p"]
        else:
            self.emu_params = emu_params

        self._build_interp()

        # train emulator
        self.train()

        # to check if emulator calls are in convex hull
        if check_hull:
            self.hull = Delaunay(self.X_param_grid)
        else:
            self.hull = None

    def _training_points_k_bin(self):
        """Method to get the Y training points in the form of the P1D
        at different k values"""

        P1D_k = np.empty([len(self.training_data), self.k_bin - 1])
        for aa in range(len(self.training_data)):
            P1D_k[aa] = self.training_data[aa]["p1d_Mpc"][1 : self.k_bin]

        return P1D_k

    def _training_points_polyfit(self):
        """Method to get the Y training points in the form of polyfit
        coefficients"""

        self._fit_p1d_in_archive(self.ndeg, self.kmax_Mpc)
        coeffs = np.empty([len(self.training_data), self.ndeg + 1])
        for aa in range(len(self.training_data)):
            coeffs[aa] = self.training_data[aa][
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

    def _buildTrainingSets(self):
        """Build the grids that contain the training parameters
        This is a nxm grid of X data (n for number of training points, m
        for number of parameters), and a length nxk set of Y  data, k being
        the number of k bins for the k bin emulator, or number of polynomial
        coefficients for the polyfit emulator"""

        ## Grid that will contain all training params
        params = np.empty([len(self.training_data), len(self.emu_params)])

        if self.emu_type == "k_bin":
            trainingPoints = self._training_points_k_bin()
        elif self.emu_type == "polyfit":
            trainingPoints = self._training_points_polyfit()
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
        """For each entry in archive, fit polynomial to log(p1d)"""

        for entry in self.training_data:
            k_Mpc = entry["k_Mpc"]
            p1d_Mpc = entry["p1d_Mpc"]
            fit_p1d = poly_p1d.PolyP1D(
                k_Mpc, p1d_Mpc, kmin_Mpc=1.0e-3, kmax_Mpc=kmax_Mpc, deg=deg
            )
            entry["fit_p1d"] = fit_p1d.lnP_fit  ## Add coeffs for each model to archive

    def _build_interp(self):
        """Method to build an GP object from a spectra archive and list of parameters
        Currently the parameter rescaling is done by taking the min and max
        of the provided params, not by defining our own prior volume. Need to decide
        whether or not this is what we want."""

        self.X_param_grid, self.Ypoints = self._buildTrainingSets()

        ## Get parameter limits for rescaling
        if self.paramLimits is None:
            self.paramLimits = self._get_param_limits(self.X_param_grid)

        ## Rescaling to unit volume
        for cc in range(len(self.training_data)):
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
            kernel = GPy.kern.Linear(len(self.emu_params), ARD=self.asymmetric_kernel)
            kernel += GPy.kern.RBF(len(self.emu_params), ARD=self.asymmetric_kernel)
        else:
            kernel = GPy.kern.RBF(len(self.emu_params), ARD=self.asymmetric_kernel)

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
        """Print the limits for each parameter"""

        for aa in range(len(self.emu_params)):
            print(self.emu_params[aa], self.paramLimits[aa])

    def return_unit_call(self, model):
        """For a given model in dictionary format, return an
        ordered parameter list with the values rescaled to unit volume
        """

        param = []
        for aa, par in enumerate(self.emu_params):
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

        param = []
        for aa, par in enumerate(self.emu_params):
            ## Rescale input parameters
            param.append(model[par])
            param[aa] = (param[aa] - self.paramLimits[aa, 0]) / (
                self.paramLimits[aa, 1] - self.paramLimits[aa, 0]
            )

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

    def emulate_p1d_Mpc(self, model, k_Mpc, return_covar=False, z=None):
        """
        Method to return the trained P(k) for an arbitrary set of k bins
        by interpolating the trained data.
        Option for reducing variance with z rescaling is not fully tested.
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
        for aa, par in enumerate(self.emu_params):
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
        for param in self.emu_params:
            model_dict[param] = self.training_data[point_number][param]

        return model_dict
