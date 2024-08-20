Installation
============

To install LaCe, these are the recommended steps:

1. Create a new conda environment and activate it. It is usually better to follow a Python version one or two behind the latest. In January 2024, the latest is 3.12, so we recommend 3.11.

   .. code-block:: bash

      conda create -n lace -c conda-forge python=3.11 camb fdasrsf pip=24.0
      conda activate lace

2. Clone the repo into your machine and perform an editable installation:

   .. code-block:: bash

      git clone https://github.com/igmhub/LaCE.git
      cd LaCE
      pip install -e .[explicit]

   If you want to use the GP emulator please run:

   .. code-block:: bash

      pip install -e .[gpy]

   If you want to use other versions of the packages you can install LaCE using:

   .. code-block:: bash

      pip install -e .

   If you want to use notebooks via JupyterHub, you'll also need to download ipykernel:

   .. code-block:: bash

      pip install ipykernel
      python -m ipykernel install --user --name lace --display-name lace

   Furthermore, the LaCE repo uses jupytext to handle notebooks and version control. To run the notebooks provided in the repo:

   .. code-block:: bash

      pip install jupytext

   .. code-block:: bash

      jupytext your_script.py --to ipynb

3. The Nyx data is not publicly available. The Nyx data is located at NERSC in:

   .. code-block:: bash

      NYX_PATH="/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files/"

   Nyx users without access to NERSC or running the code elsewhere need to add the Nyx path as an environment variable in the notebook kernel. This is done by writing in the kernel.json file:

   .. code-block:: json

      "env": {
         "NYX_PATH":"path_to_Nyx"
      }

4. To improve the reading time, you can precompute all cosmological information needed using CAMB. This is done by running the script:

   .. code-block:: bash

      python scripts/compute_nyx_emu_cosmo.py

   Note that you may need to update it as explained inside the file.
