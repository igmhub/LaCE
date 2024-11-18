
## Installation

We recommend using conda environment to install LaCE. To create a conda environment, one needs to specify the python version.
In January 2024, the latest is 3.12. To use the emulators, there are no strong requirements on the python version. LaCE also contains a cosmology module to estimate compressed cosmological parameters using cosmopower, and this module requires python 3.10. 

The following instructions are for python 3.10 to ensure that all modules in the repository can run.

To create a conda environment with python 3.10, run:

```
conda create -n lace -c conda-forge python=3.10 camb fdasrsf cosmopower, pyDOE2
conda activate lace
pip install --upgrade pip
```

- Clone the repo into your machine and perform an *editable* installation:

```
git clone https://github.com/igmhub/LaCE.git
cd LacE
pip install -e .
``` 
