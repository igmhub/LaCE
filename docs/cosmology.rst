Cosmology
=========

``lace.cosmo`` provides the shared cosmology interface used by LaCE and its
downstream analyses. ``Cosmology`` evaluates the requested cosmology with
CAMB. ``RescaledCosmology`` reuses a fiducial cosmology when the background
parameters are fixed and only primordial linear-power parameters change.

Use ``RescaledCosmology`` for changes in ``As``, ``ns``, or ``nrun`` with the
same background. Analyses that change the background should construct a new
``Cosmology`` object so CAMB recomputes the expansion history and transfer
function. ``notebooks/cosmology/compare_rescaled_cosmology.py`` demonstrates
the fixed-background comparison against fresh CAMB calculations, including the
``Delta2_star``, ``n_star``, and ``alpha_star`` parameters used by cup1d.

The API reference documents the available linear-power and compressed-parameter
methods.
