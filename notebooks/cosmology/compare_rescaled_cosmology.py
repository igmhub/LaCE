# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rescaled cosmology against fresh CAMB calculations
#
# `RescaledCosmology` adjusts primordial parameters while keeping the fiducial
# expansion and transfer function. Here we compare independent CAMB calculations
# for changes in `As` (primordial amplitude), `ns` (spectral tilt), and `nrun`
# (running). We also check that changing `H0` requires a new cosmology.
#
# The automated counterpart is `tests/test_rescaled_cosmology.py`; run it with
# `pytest -q tests/test_rescaled_cosmology.py`.

# %%
import numpy as np

from lace.cosmo.cosmology import Cosmology
from lace.cosmo.rescale_cosmology import RescaledCosmology
from lace.plotting.cosmology import plot_ratio_curves

z_target = 2.33  # example target redshift for emulator parameters
z_star = 3.0  # fixed redshift for cup1d star parameters
k_Mpc = np.geomspace(0.1, 4.0, 100)  # comoving wavenumber [1/Mpc]
kp_Mpc = 0.7  # emulator compressed-parameter pivot [1/Mpc]
kp_kms = 0.009  # cup1d velocity-space pivot [s/km]
changes = {"As": {"As": 2.2e-9}, "ns": {"ns": 0.96}, "nrun": {"nrun": -0.01}}
fiducial = Cosmology()

# %% [markdown]
# ## Fixed-background linear power
#
# Both methods return linear CDM+baryon power `P(k,z)` in `Mpc³`. The same
# `k` values and target redshift are used for each comparison. Require agreement to
# `rtol=1e-4` (0.01%); this permits small CAMB interpolation differences.

# %%
power_ratios = {}
for name, parameters in changes.items():
    rescaled = RescaledCosmology(fiducial, parameters)
    fresh = Cosmology(cosmo_params_dict=parameters)
    ratio = rescaled.get_linP_Mpc(z_target, k_Mpc) / fresh.get_linP_Mpc(z_target, k_Mpc)
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-4)
    print(f"{name}: maximum |P_rescaled/P_CAMB - 1| = {np.max(np.abs(ratio - 1)):.3g}")
    power_ratios[name] = ratio - 1
plot_ratio_curves(
    k_Mpc,
    power_ratios,
    ylabel=r"$P_\mathrm{rescaled}/P_\mathrm{CAMB}-1$",
)

# %% [markdown]
# ## Emulator parameters at the target redshift
#
# The preceding cell compared the full linear spectra from `RescaledCosmology`
# and a fresh CAMB `Cosmology`. Here we compare their **compressed descriptions
# of those spectra**. For each change in `As`, `ns`, or `nrun`, the two methods
# should give the same `Delta2_p`, `n_p`, and `alpha_p` at `z_target=2.33` and
# `kp_Mpc=0.7 1/Mpc`. These are the dimensionless amplitude, logarithmic slope,
# and curvature used by the emulator. Each pair is compared within
# `rtol=1e-4, atol=1e-6`.

# %%
for name, parameters in changes.items():
    rescaled = RescaledCosmology(fiducial, parameters)
    fresh = Cosmology(cosmo_params_dict=parameters)
    rescaled_p = rescaled.get_linP_Mpc_params(z_target, kp_Mpc)
    camb_p = fresh.get_linP_Mpc_params(z_target, kp_Mpc)
    for key in ("Delta2_p", "n_p", "alpha_p"):
        np.testing.assert_allclose(rescaled_p[key], camb_p[key], rtol=1e-4, atol=1e-6)
        print(
            name,
            key,
            rescaled_p[key],
            "rescaling/direct ratio:",
            rescaled_p[key] / camb_p[key],
        )

# %% [markdown]
# ## Star parameters at the fixed redshift
#
# This is a **separate** comparison between the same two methods. At
# `z_star=3`, cup1d uses the velocity-space pivot `kp_kms=0.009 s/km` to define
# `Delta2_star`, `n_star`, and `alpha_star`. We compare each value from
# `RescaledCosmology` against the corresponding value from fresh CAMB. We do
# not compare `_star` with `_p`: their redshifts and pivots differ.
#
# LaCE implements the velocity-space fit by converting its pivot to comoving
# units using `k_Mpc = k_kms H(z)/(1+z)`, then fitting `P_Mpc`. At fixed
# redshift this gives the same dimensionless amplitude, slope, and curvature
# as fitting `P_kms`, because the power conversion is a constant factor.
# The code also checks the conversion factor `H(z)/(1+z)` in `km/s/Mpc` and
# linear power in velocity units, `(km/s)³`. The parameter comparison uses
# `rtol=1e-4, atol=1e-6`.

# %%
for name, parameters in changes.items():
    rescaled = RescaledCosmology(fiducial, parameters)
    fresh = Cosmology(cosmo_params_dict=parameters)
    rescaled_star = rescaled.get_linP_kms_params(z_star, kp_kms)
    camb_star = fresh.get_linP_kms_params(z_star, kp_kms)
    for key in ("Delta2_star", "n_star", "alpha_star"):
        np.testing.assert_allclose(rescaled_star[key], camb_star[key], rtol=1e-4, atol=1e-6)
        print(
            name,
            key,
            rescaled_star[key],
            "rescaling/direct ratio:",
            rescaled_star[key] / camb_star[key],
        )
    np.testing.assert_allclose(rescaled.get_dkms_dMpc(z_star), fresh.get_dkms_dMpc(z_star), rtol=1e-6)
    np.testing.assert_allclose(rescaled.get_linP_kms(z_star, kp_kms), fresh.get_linP_kms(z_star, kp_kms), rtol=1e-4)

# %% [markdown]
# ## Changed background
#
# `H0` is in `km/s/Mpc`. `RescaledCosmology` must reject a changed background;
# a fresh `Cosmology` then runs CAMB with the new expansion history.

# %%
background_change = {"H0": 74.0}
assert not fiducial.same_background(background_change)
try:
    RescaledCosmology(fiducial, background_change)
except AssertionError as error:
    assert str(error) == "background not fixed"
else:
    raise AssertionError("RescaledCosmology accepted a changed background")

fresh_background = Cosmology(cosmo_params_dict=background_change)
new_conversion = fresh_background.get_dkms_dMpc(z_star)
fiducial_conversion = fiducial.get_dkms_dMpc(z_star)
assert not np.isclose(new_conversion, fiducial_conversion, rtol=1e-4)
print("Velocity-to-comoving conversion ratio at z_star:", new_conversion / fiducial_conversion)

new_power = fresh_background.get_linP_Mpc(z_target, k_Mpc)
fiducial_power = fiducial.get_linP_Mpc(z_target, k_Mpc)
assert not np.allclose(new_power, fiducial_power, rtol=1e-4)
print("P_new/P_fid at z_target, k=0.7 1/Mpc:", fresh_background.get_linP_Mpc(z_target, np.array([kp_Mpc]))[0] / fiducial.get_linP_Mpc(z_target, np.array([kp_Mpc]))[0])

# %%

# %%
