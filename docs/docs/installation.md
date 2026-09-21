# Installation

LaCE requires Python 3.12 or newer. Create an environment, install LaCE, and
then install any optional tools required by the workflow you intend to use.

```
conda create -n lace python=3.12
conda activate lace
python -m pip install --upgrade pip
git clone https://github.com/igmhub/LaCE.git
cd LaCE
python -m pip install -e .
```

The Nyx archive requires the ``NYX_PATH`` environment variable to point to the
directory containing the Nyx data files. See the repository README for the
NERSC path and commands that prepare the required cosmology and IGM assets.
