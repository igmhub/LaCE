# LaCE

LaCE contains a set of emulators for the one-dimensional flux power spectrum of the Lyman-alpha forest. It has been used in the papers:

- https://arxiv.org/abs/2011.15127
- https://arxiv.org/abs/2209.09895
- https://arxiv.org/abs/2305.19064 (latest version)

Please cite at least https://arxiv.org/abs/2305.19064 if you use this emulator in your research.

## Installation
(Last updated: Jan 19 2024)

- Create a new conda environment. It is usually better to follow python version one or two behind. In January 2024, the latest is 3.12, so we recommend 3.11. If you want to use LaCE with cosmopower, as of November 2025 you need to install python 3.10. Please look at the cosmopower installation before proceeding with the LaCE installation.

```
conda create -n lace -c conda-forge python=3.11 camb fdasrsf
conda activate lace
pip install --upgrade pip
```

- Clone the repo into your machine and perform an *editable* installation:

```
git clone https://github.com/igmhub/LaCE.git
cd LacE
pip install -e .
``` 

- If you find problems, please install LaCE as follows:

```
pip install -e .[explicit]
```

- If you want to use cosmopower, you need to use python 3.10 and install the cosmopower package. You must install cosmopower before installing LaCE. You can do this by running:

```
pip install cosmopower
```
and install:
```
pip install -e .
pip install pyDOE2
pip install git+https://github.com/igmhub/cup1d.git
```

#### Tests

Please run the following script to check that the package is working properly.

```
=======
#### Tests

Please run the following script to check that the package is working properly.

```
python test_lace.py
```

#### Nyx users:

- You may need to add the Nyx path as an environment variable in your notebook kernel. The first is done by writing in the kernel.json file:

```
 "env": {
  "NYX_PATH":"path_to_Nyx"
 }
```

You also need to add the Nyx path as an environment variable. The Nyx data is located at NERSC in 

```
NYX_PATH="/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files/"
```

- Before running LaCE, please precompute all cosmological information needed using CAMB and save IGM histories. This is done by running the following scripts. You do not need to do it if you are in NERSC.

```
python scripts/save_nyx_emu_cosmo.py
python scripts/save_nyx_IGM.py
```


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


### Notebooks / tutorials


- All notebooks in the repository are in .py format. To generate the .ipynb version, run:

```
jupytext --to ipynb notebooks/*.py
```

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel`:

```
pip install ipykernel
python -m ipykernel install --user --name cup1d --display-name lace
```
In the `Notebooks` folder, there are several tutorials one can run to learn how to use the archives and emulators.

- Archive tutorial: notebooks/Tutorial_bookkeeping.py
- Emulator tutorial: notebooks/Tutorial_emulator.py
- Emulating compressed parameters tutorial: notebooks/Tutorial_compressedParams.ipynb
