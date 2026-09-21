Tutorials
=========

Notebook sources are maintained as paired Jupytext Python files and Jupyter
notebooks. The main practical examples are in ``notebooks``:

``notebooks/tutorials/Tutorial_bookkeeping.py``
   Inspect simulation archive contents and parameter bookkeeping.

``notebooks/tutorials/Tutorial_emulator.py``
   Load and evaluate a P1D emulator.

``notebooks/compressed_parameters/Tutorial_compressedParams.py``
   Compute compressed linear-power parameters from a cosmology.

``notebooks/archive/compute_covariance.py``
   Train leave-one-out GP emulators and inspect their covariance behavior.

Synchronize a paired notebook after editing its Python source with:

.. code-block:: console

   jupytext --sync notebooks/tutorials/Tutorial_emulator.py
