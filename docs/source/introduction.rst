Introduction
============

Welcome to the LaCE documentation! LaCE contains a set of emulators for the one-dimensional flux power spectrum of the Lyman-alpha forest. It has been used in the papers:

- `arxiv:2011.15127 <https://arxiv.org/abs/2011.15127>`_
- `arxiv:2209.09895 <https://arxiv.org/abs/2209.09895>`_
- `arxiv:2305.19064 (latest version) <https://arxiv.org/abs/2305.19064>`_

Please cite at least `arxiv:2305.19064 <https://arxiv.org/abs/2305.19064>`_ if you use this emulator in your research.

The LaCE emulators comprise a set of emulators that parameterize cosmological effects using the linear matter power spectrum.
LaCE supports two algorithm options:

- Neural Networks (LaCE-NN) :ref:`LaCE-NN <nn_emulator>`.

- Gaussian Process (LaCE-GP)  :ref:`LaCE-GP <gp_emulator>`.

And two suites of hydrodynamical simulations:

- **MP-Gadget simulations:** These were conducted using the `Gadget` code, a Smoothed Particle Hydrodynamics (SPH) simulation framework that employs a Lagrangian method to represent the simulation domain as a collection of particles. Each \texttt{MP-Gadget} simulation box has a size of :math:`L = 67.5` Mpc on each side. The simulations generate 11 output snapshots uniformly spaced in redshift between :math:`z = 4.5` and :math:`z = 2`. This makes a total of 330 training points. We have also calculated four additional mean-flux rescalings per snapshot, which enlarge the dataset to 1650 training points.

- **Nyx simulations:** [lacking documentation here still, WiP]

The LaCE emulators comprise a set of emulators that parameterize cosmological effects using the linear matter power spectrum. However, the Lyman-:math:`\alpha` forest is influenced by both cosmological parameters and the thermal state of the IGM. Consequently, the LaCE emulators incorporate four parameters describing IGM physics that characterize the :math:`P_{\rm 1D}`.

- Cosmological parameters:
  - :math:`\Delta^2_p` is the amplitude of the (dimensionless) linear spectrum at a pivot scale :math:`k_p = 0.7` 1/Mpc
  - :math:`n_p` is the slope of the linear power spectrum at :math:`k_p`
  - :math:`\alpha_p` is the running of the linear power spectrum at :math:`k_p`
  - :math:`f_p` is the (scale-independent) logarithmic growth rate

- IGM parameters:
  - :math:`\bar{F}` is the mean transmitted flux fraction in the box (mean flux), which encodes information about the ionization state of the gas and is related to the effective optical depth as :math:`\tau_{\rm eff} = -\log \bar{F}`.
  - :math:`\sigma_T` [Mpc] is the thermal broadening scale in comoving units, computed from :math:`T_0` in the temperature-density relation
  - :math:`\gamma` is the slope of the temperature-density relation
  - :math:`k_F` [Mpc] is the filtering length (or pressure smoothing scale) in inverse comoving units

To mitigate the impact of cosmic variance, LaCE emulates a smoothed version of the :math:`P_{\rm 1D}`. Currently, this smoothing is achieved using polynomials.

LaCE provides a set of stable emulator versions. An emulator version comprises both an archive (Gadget or Nyx) and an emulator (GP, NN) optimized for such archive. As of Aug 5th 2024, the recommended versions are:

- **Pedersen21_ext:** GP emulator used in `arxiv:2011.15127 <https://arxiv.org/abs/2011.15127>`_ and `arxiv:2209.09895 <https://arxiv.org/abs/2209.09895>`_. It uses the Gadget archive without mean-flux rescalings (330 training points). It emulates to scales of :math:`4 \ \text{Mpc}^{-1}` and uses [:math:`\Delta^2_p`, :math:`n_p`, :math:`\bar{F}`, :math:`\sigma_T`, :math:`\gamma`, :math:`k_F`].

- **Pedersen21_ext8:** Equivalent to Pedersen21_ext, but accessing smaller scales of :math:`8 \ \text{Mpc}^{-1}`.

- **Cabayol23+:** NN emulator used in `arxiv:2305.19064 <https://arxiv.org/abs/2305.19064>`_. It includes some minor improvements with respect to the version in the paper. It uses the Gadget archive with mean-flux rescalings. It emulates to scales of :math:`4 \ \text{Mpc}^{-1}` and uses [:math:`\Delta^2_p`, :math:`n_p`, :math:`\bar{F}`, :math:`\sigma_T`, :math:`\gamma`, :math:`k_F`].

- **Cabayol23+_extended:** Equivalent to Cabayol23+, but accessing smaller scales of :math:`8 \ \text{Mpc}^{-1}`.

- **Nyx_alphap:** NN emulator using the Nyx archive. It uses the common LaCE emulator parameters and additionally incorporates :math:`\alpha_p`. It emulates to scales of :math:`4 \ \text{Mpc}^{-1}` and uses [:math:`\Delta^2_p`, :math:`n_p`, :math:`\alpha_p`, :math:`\bar{F}`, :math:`\sigma_T`, :math:`\gamma`, :math:`k_F`].

Currently, the GP option on the Nyx archive is not supported.
