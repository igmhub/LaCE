# LaCE

LaCE contains a set of emulators for the one-dimensional flux power spectrum of the Lyman-alpha forest. It has been used in the papers:

- https://arxiv.org/abs/2011.15127
- https://arxiv.org/abs/2209.09895
- https://arxiv.org/abs/2305.19064 
- https://arxiv.org/abs/2601.21432 (latest version)

Please cite at least https://arxiv.org/abs/2601.21432 if you use this emulator in your research.

There is a documentation website with installation instructions and relevant descriptions at [here](https://igmhub.github.io/LaCE/

## Installation
(Last updated: Jul 2 2026)

- Create a new conda environment

```
conda create -n lace python=3.12
conda activate lace
pip install --upgrade pip
```

- Clone the repo into your machine and perform an *editable* installation:

```
git clone https://github.com/igmhub/LaCE.git
cd LaCE
pip install -e .
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
python scripts/developers/save_nyx_emu_cosmo.py
python scripts/developers/save_nyx_IGM.py
```

## Emulator parameters:

These are the parameters that describe each individual P1D(kpar) power spectrum. We have detached these from redshift and traditional cosmology parameters.

#### Cosmological parameters:

`Delta2_p` is the amplitude of the (dimensionless) linear spectrum at k_p = 0.7 1/Mpc

`n_p` is the slope of the linear power spectrum at k_p


#### IGM parameters:

`mF` is the mean transmitted flux fraction in the box (mean flux)

`sigT_Mpc` is the thermal broadening scale in comoving units, computed from `T_0` in the temperature-density relation

`gamma` is the slope of the temperature-density relation

`kF_Mpc` is the filtering length (or pressure smoothing scale) in inverse comoving units


## Notebooks / tutorials


- All notebooks in the repository are in .py format. To generate the .ipynb version, run:

```
jupytext --to ipynb notebooks/*.py
```

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel`:

```
pip install ipykernel
python -m ipykernel install --user --name lace --display-name lace
```
In the `Notebooks` folder, there are several tutorials one can run to learn how to use the archives and emulators.

- Archive tutorial: notebooks/Tutorial_bookkeeping.py
- Emulator tutorial: notebooks/Tutorial_emulator.py
- Emulating compressed parameters tutorial: notebooks/Tutorial_compressedParams.ipynb
