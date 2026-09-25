End-to-end workflow
===================

This workflow loads simulation measurements, selects emulator inputs, loads a
production emulator, predicts P1D, and checks the domain in which that
prediction is supported.

Data flow
---------

.. graphviz::

   digraph lace_workflow {
       graph [rankdir=LR, bgcolor="transparent"];
       node [shape=box, style="rounded,filled", fillcolor="#eef4fb"];
       Files [label="simulation\npost-processing"];
       Archive [label="GadgetArchive /\nNyxArchive"];
       Samples [label="training or\ntesting snapshots"];
       Emulator [label="set_emulator"];
       Prediction [label="P1D_Mpc"];
       Validation [label="training range +\nL1O residuals"];
       Files -> Archive -> Samples;
       Samples -> Emulator [label="parameter mapping"];
       Emulator -> Prediction [label="k_iMpc"];
       Samples -> Validation;
       Prediction -> Validation;
   }

1. Load an archive
------------------

The public MP-Gadget products are available through ``GadgetArchive``.
Nyx users must configure their external data path as described in
:doc:`installation`.

.. code-block:: python

   from lace.archive.gadget_archive import GadgetArchive

   archive = GadgetArchive(postproc="Cabayol23")
   emulator_parameters = [
       "Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"
   ]
   training = archive.get_training_data(
       emulator_parameters, average="both"
   )
   testing = archive.get_testing_data("mpg_central")

Each snapshot is a mapping containing redshift, cosmology, IGM state, and its
measured P1D. Archive files retain historical keys such as ``k_Mpc`` and
``p1d_Mpc``; new prediction APIs use ``k_iMpc`` and ``P1D_Mpc``.

2. Load a production emulator
-----------------------------

.. code-block:: python

   from lace.emulator import set_emulator

   emulator = set_emulator("CH24_mpgcen_gpr")

Use ``CH24_nyxcen_gpr`` for the supported Nyx-trained alternative. An
installed package can load an external trusted model bundle with
``model_path=``; see :doc:`emulators`.

3. Evaluate P1D
---------------

An archive snapshot already contains the complete parameter mapping expected
by the matching emulator:

.. code-block:: python

   import numpy as np

   k_iMpc = np.geomspace(0.1, 4.0, 100)
   P1D_Mpc = emulator.emulate_P1D_Mpc(testing[0], k_iMpc)

For a batch, pass arrays under each emulator-parameter key. The returned power
has wavenumber on the final axis. Production GP covariance is distributed as
a separate leave-one-out validation product rather than inferred from this
prediction call.

4. Check validity and uncertainty
---------------------------------

Before evaluating arbitrary parameters, derive their sampled range from
``training``:

.. code-block:: python

   bounds = {
       name: (
           min(sample[name] for sample in training),
           max(sample[name] for sample in training),
       )
       for name in emulator_parameters
   }

These component-wise bounds are a useful diagnostic, not proof that a point
lies inside the multivariate training domain. Predictions outside the
training distribution are extrapolations.

Emulator covariance is estimated from leave-one-simulation-out residuals. It
describes model error on held-out simulations and is distinct from simulation
sample variance and observational covariance. Keep these sources separate
until the downstream likelihood explicitly combines them.

5. Continue downstream
----------------------

LaCE returns a comoving ``P1D_Mpc``. cup1d converts it to
``P1D_kms``, applies contaminants and systematics, and compares it with
data. ForestFlow instead predicts Arinyo coefficients before constructing P3D
and P1D. See :doc:`tutorials` for executable examples and
:doc:`conventions` for units and array contracts.
