# Installation
(Last updated: Nov 19 2024)

LaCE contains a submodule to estimate compressed parameters from the power spectrum that uses cosmopower. The LaCE installation is slightly different depending on whether you want to use cosmopower or not.

### LaCE without cosmopower

- Create a new conda environment. It is usually better to follow python version one or two behind. In January 2024, the latest is 3.12, so we recommend 3.11. If you want to use LaCE with cosmopower, as of November 2025 you need to install python 3.10. Please look at the cosmopower installation before proceeding with the LaCE installation.

```
conda create -n lace -c conda-forge python=3.11 pip 
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
pip install -e ".[explicit]"
```

### LaCE with cosmopower

- Create a new conda environment. 

```
conda create -n lace -c conda-forge python=3.11 pip 
conda activate lace
pip install --upgrade pip
```
- Install cosmopower:

```
pip install cosmopower pyDOE
```

- Clone the repo into your machine and perform an *editable* installation:

```
git clone https://github.com/igmhub/LaCE.git
cd LacE
``` 
- Install LaCE using the installation with explicit dependencies:
```
pip install -e ".[explicit]"
```