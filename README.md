# LaCE

LaCE contains a set of emulators for the one-dimensional flux power spectrum of the Lyman-alpha forest. It has been used in the papers:

- https://arxiv.org/abs/2011.15127
- https://arxiv.org/abs/2209.09895
- https://arxiv.org/abs/2305.19064 
- https://arxiv.org/abs/2601.21432 (latest version)

Please cite at least https://arxiv.org/abs/2601.21432 if you use this emulator in your research.

The documentation covers installation, cosmology, archives, emulators, and
the package API:

- [Online documentation](https://igmhublace.readthedocs.io/en/latest/)
- [Documentation source](https://github.com/igmhub/LaCE/tree/main/docs)

Build it locally with:

```bash
python -m pip install -e ".[docs]"
make docs
```

Open `docs/_build/html/index.html` after the build completes.

## Installation
(Last updated: Jul 27 2026)

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
make install
``` 

#### Nyx users:

LaCE uses `/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files/`
as its default Nyx directory at NERSC. On another machine, set your local
directory once from the command line:

```bash
set_nyx_path /path/to/nyx_files
```

This writes `~/.config/lace/paths.toml`. You can also pass `nyx_path=`
directly to `NyxArchive` for a one-off location. The equivalent Python call is
`lace.configuration.set_nyx_path("/path/to/nyx_files")`.

- Before running LaCE, please precompute all cosmological information needed using CAMB and save IGM histories. This is done by running the following scripts. You do not need to do it if you are in NERSC.

```
python scripts/developers/save_nyx_emu_cosmo.py
python scripts/developers/save_nyx_IGM.py
```

### Running tests

Run the complete test suite with:

```bash
make test
```

The suite checks the Pedersen21 archive, default cosmology reference values,
the packaged MP-Gadget emulator, and star parameters from
`RescaledCosmology` against fresh CAMB calculations.

### External model and archive assets

LaCE keeps simulation suites and L1O weights outside the package. Configure an external root containing `GPmodels`, `ff_mpgcen.npy`, and optionally `sim_suites` with `lace.configuration.set_data_path("/path/to/lace-data")`, or pass `data_path`/`model_path` to the archive or supported emulator factory. Explicit arguments take precedence over persistent configuration. See the emulator documentation for trusted-pickle, manifest, compatibility, and L1O guidance.

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


- Notebook sources are maintained as Jupytext Python files. After `make install`, generate or refresh every notebook with:

```
make notebooks
```

Run this from the LaCE repository root. The target searches only `LaCE/notebooks/`, recursively, and skips Jupyter checkpoint files.

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel`:

```
pip install ipykernel
python -m ipykernel install --user --name lace --display-name lace
```
In the `Notebooks` folder, there are several tutorials one can run to learn how to use the archives and emulators.

- [Archive tutorial](notebooks/tutorials/Tutorial_bookkeeping.ipynb)
- [Emulator tutorial](notebooks/tutorials/Tutorial_emulator.ipynb)
- [Compressed-parameter tutorial](notebooks/cosmology/cosmopower/Tutorial_compressedParams.ipynb)

For a complete archive-to-prediction example, see the
[end-to-end workflow](https://igmhublace.readthedocs.io/en/latest/workflow.html).

### Versioning

Package versions are derived from Git. Tagged releases use the tag; development builds include the commit distance and short SHA (for example, `1.2.0.dev4+gabc1234`). A dirty working tree adds `.dirty`. Source archives without Git metadata report `0+unknown`.
