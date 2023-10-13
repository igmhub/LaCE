# LaCE

Lyman-alpha Cosmology Emulator. This code is a Gaussian process emulator for the 1D flux power spectrum
of the Lyman-alpha forest, and was used to generate the results shown in
https://arxiv.org/abs/2011.15127. Please cite this article if you use this emulator in your research.

## Installation
(Instructions by Laura Cabayol on July 21st 2023. Updated by Naim Karacayli on Oct 12, 2023)

- To install on NERSC, you first need to load python module with `module load python`. This is not necessary for personal computers.

- Create a new conda environment. It is usually better to follow python version one or two behind. In October 2023, latest is 3.11, so we recommend 3.10.

```
conda create -n lace python=3.10
conda activate lace
```

- If you are **not** a developer, simply pip install. The following command will take care of the dependencies.

`pip install git@github.com:igmhub/LaCE.git`
    
- If you are a developer, first clone the repo into your machine and perform an *editable* installation:

```
git clone git@github.com:igmhub/LaCE.git
cd LacE
pip install -e .
``` 

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel`:

```
pip install ipykernel
python -m ipykernel install --user --name lace --display-name lace
```


- You may need to add the Nyx path as an enviroment variable in your notebook kernel. Also, if you want to use the Nyx archive, the path also needs to be added as an environment variable.
This is done writting in the kernel.json file:

```
 "env": {
  "NYX_PATH":"path_to_Nyx"
 }
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


#### Tutorials:

In the `Notebooks` folder, there are several tutorials one can run to learn how to use
the emulators and archives. The `Durham2023_LaCETutorial` is the more complete one that
shows how to get the different archives and run them with the available emulator options.


