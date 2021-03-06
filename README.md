# LaCE

Lyman-alpha Cosmology Emulator. This code is a Gaussian process emulator for the 1D flux power spectrum
of the Lyman-alpha forest, and was used to generate the results shown in
https://arxiv.org/abs/2011.15127. Please cite this article if you use this emulator in your research.

## Installation

Set an environment variable `export LACE_REPO=/path/to/repo/LaCE`. This will be needed to use the code, so its best to have this
in something like a `.bashrc` so it is always defined.

### Dependencies:
Python version 3.6 or later is necessary due to `CAMB` version dependencies.

The following modules are required:

`numpy`

`scipy`

`matplotlib`

`configobj`

`CAMB` version 1.1.3 or later https://github.com/cmbant/CAMB (only works with Python 3.6 or later as of 14/01/2021)

`GPy` (only works with Python 3.8 or lower, not compatible with 3.9 as of 14/01/2021)

`classylss` (not at this point, I think)

### Installation at NERSC

(Instructions by Andreu Font-Ribera on March 21st 2022)

On a fresh terminal:

`module load python`

`conda create -n lace_env python=3.8 pip`

`source activate lace_env`

`pip install gpy configobj matplotlib`

`pip install camb`

Followed by:

`git clone git@github.com:igmhub/LaCE.git`

`cd LaCE`

`python setup.py install`

If you want to use notebooks via JupyterHub, you'll also need:

`pip install ipykernel`

`python -m ipykernel install --user --name lace_env --display-name lace_env`


## Emulator parameters:

These are the parameters that describe each individual P1D(k) power spectrum. We have detached these from redshift and traditional cosmology parameters.

#### Cosmological parameters:

`Delta2_p` is the amplitude of the (dimensionless) linear spectrum at k_p = 0.7 1/Mpc

`n_p` is the slope of the linear power spectrum at k_p

`alpha_p` is the running of the linear power spectrum at k_p

`f_p` is the (scale-independent) logarithmic growth rate

The current version of the emulator, relased in this repo, does not emulate `alpha_p` and `f_p`. However, these parameters are stored in the P1D archive.

#### IGM parameters:

`mF` is the mean transmitted flux fraction in the box (mean flux)

`sigT_Mpc` is the thermal broadening scale in comoving units, computed from `T_0` in the temperature-density relation

`gamma` is the slope of the temperature-density relation

`kF_Mpc` is the filtering length (or pressure smoothing scale) in inverse comoving units


## Saving and loading emulator hyperparameters

The default operation of the emulator is currently to optimise a new set of hyperparameters on whichever training set it is initialised with. However, one can also run with the `train=False` flag, and use GPEmulator.load_default(). This will load a standardised set of hyperparameters (along with the appropriate parameter rescalings for the X training data) that are optimised on the entire suite.

