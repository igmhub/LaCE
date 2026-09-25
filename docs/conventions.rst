Scientific conventions
======================

Names carry units
-----------------

All dimensional quantities in public LaCE interfaces include their unit in
the name. The letter ``i`` means *inverse*: ``k_iMpc`` has units
Mpc^-1 and ``k_ikms`` has units (km/s)^-1 = s/km. Positive powers do not
use ``i``: ``P1D_Mpc`` has units Mpc, ``P3D_Mpc`` has units Mpc^3,
and ``P1D_kms`` has units km/s.

The velocity/comoving conversion is named ``dkms_diMpc`` and has units
(km/s)/Mpc. Consequently,

.. math::

   k_{iMpc} = k_{ikms}\,dkms_{diMpc},\qquad
   P1D_{kms} = P1D_{Mpc}\,dkms_{diMpc}.

Capitalization is part of the convention: public power-spectrum names use
``P1D`` and ``P3D``. Dimensionless quantities such as ``z``, ``mu``,
``mF``, ``Delta2_star`` and ``n_star`` have no unit suffix.

Arrays and emulator outputs
---------------------------

Wavenumber arrays supplied to public prediction methods are finite and
positive. A single model and a one-dimensional ``k_iMpc`` array produce a
one-dimensional ``P1D_Mpc`` array of equal length. Batched interfaces place
the model/redshift axis first and the wavenumber axis last. A covariance of
``N`` returned P1D values has shape ``(N, N)`` and units Mpc squared.

Legacy names
------------

Existing archives using ``k_Mpc``, ``p1d_Mpc`` or ``dkms_dMpc`` remain
readable. These spellings and ``emulate_p1d_Mpc`` are compatibility aliases;
new public code must use ``k_iMpc``, ``P1D_Mpc``, ``dkms_diMpc`` and
``emulate_P1D_Mpc``.
