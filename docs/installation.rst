Installation
============

LaCE requires Python 3.12 or newer. It is normally installed alongside its
scientific dependencies in a dedicated environment:

.. code-block:: console

   conda create -n lace python=3.12
   conda activate lace
   git clone https://github.com/igmhub/LaCE.git
   cd LaCE
   python -m pip install -e .

Install the documentation tools and build the site locally with:

.. code-block:: console

   python -m pip install -e ".[docs]"
   make docs

Nyx archive users must set ``NYX_PATH`` to the directory containing the Nyx
data files. The repository README describes the NERSC location and the helper
scripts used to prepare cosmology and IGM assets.
