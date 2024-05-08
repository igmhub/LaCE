# LaCE

LaCE contains a set of emulators for the one-dimensional flux power spectrum of the Lyman-alpha forest. It has been used in the papers:

- https://arxiv.org/abs/2011.15127
- https://arxiv.org/abs/2209.09895
- https://arxiv.org/abs/2305.19064 (latest version)

Please cite at least https://arxiv.org/abs/2305.19064 if you use this emulator in your research.

## Installation
(Last update Jan 19 2024)

- Create a new conda environment. It is usually better to follow python version one or two behind. In January 2024, the latest is 3.12, so we recommend 3.11.

```
conda create -n lace -c conda-forge python=3.11
conda activate lace
```

- Clone the repo into your machine and perform an *editable* installation:

```
git clone https://github.com/igmhub/LaCE.git
cd LacE
pip install -e .
``` 

- If you want to fixed versions for dependencies, run ``pip install -e .[explicit]`` instead.

  
- If you want to use the GP emulator please run:


```
pip install -e .[gpy]
``` 

- If you want to use other versions of the packages you can install LaCE using:

```
pip install -e .
``` 

but there may be some issues.


- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel`:

```
pip install -e .[jupyter]
pip install ipykernel
python -m ipykernel install --user --name lace --display-name lace
```

#### Nyx users:

- You may need to add the Nyx path as an enviroment variable in your notebook kernel. The first is done writting in the kernel.json file:

```
 "env": {
  "NYX_PATH":"path_to_Nyx"
 }
```

For convenience, you could also add the Nyx path as an environment variable.

- To improve the reading time, you can precompute all cosmological information needed using CAMB. This is done by running the script 

```
python scripts/compute_nyx_emu_cosmo.py
```

Note that you may need to update it as explained inside the file.


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


#### Tutorials:

In the `Notebooks` folder, there are several tutorials one can run to learn how to use the archives and emulators.

- Archive tutorial: notebooks/Tutorial_bookkeeping.ipynb
- Emulator tutorial: notebooks/Tutorial_emulator.ipynb

In the `tests` folder, there are four scripts that you can run to open a Gadget or Nyx archive and run the G^and NN emulators.
