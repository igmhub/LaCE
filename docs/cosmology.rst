Cosmology
=========

``lace.cosmo`` provides the shared cosmology interface used by LaCE and its
downstream analyses. ``Cosmology`` evaluates the requested cosmology with
CAMB. ``RescaledCosmology`` reuses a fiducial cosmology when the background
parameters are fixed and only primordial linear-power parameters change.

Use ``RescaledCosmology`` for changes in ``As``, ``ns``, or ``nrun`` with the
same background. Analyses that change the background should construct a new
``Cosmology`` instance so CAMB can recompute the transfer functions.

The API reference documents the available linear-power and compressed-parameter
methods.
