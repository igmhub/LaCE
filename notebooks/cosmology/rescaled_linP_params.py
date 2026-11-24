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
# # Exact rescaling of the compressed linear-power parameters
#
# `get_linP_Mpc_params` fits a quadratic in $x=\ln(k/k_p)$ to $\ln P_\mathrm{lin}$
# on a fixed grid ($0.5\,k_p$–$2\,k_p$) and returns
#
# $$\ln P \simeq \ln A_p + n_p\,x + \tfrac{1}{2}\alpha_p\,x^2,\qquad
# \Delta^2_p = A_p\,k_p^3/(2\pi^2).$$
#
# With a fixed background, `RescaledCosmology` multiplies the fiducial power by
# the primordial ratio, $\ln S = a + b L + c L^2 + d L^3$ with $L=\ln(k/k_s)=x+\ell$,
# $\ell=\ln(k_p/k_s)$, $a=\ln(A_s/A_s^\mathrm{fid})$, $b=\Delta n_s$,
# $c=\Delta n_\mathrm{run}/2$, $d=\Delta n_\mathrm{runrun}/6$. Since the fit is
# linear in $\ln P$ and $\ln S$ is a polynomial in $x$:
#
# $$\ln\Delta^2_p = \ln\Delta^2_{p,\mathrm{fid}} + a + b\ell + c\ell^2,\qquad
# n_p = n_{p,\mathrm{fid}} + b + 2c\ell,\qquad
# \alpha_p = \alpha_{p,\mathrm{fid}} + 2c$$
#
# (plus a fixed projection of $x^3$ when $n_\mathrm{runrun}$ changes). The
# fiducial fit is done once per $(z, k_p)$ and cached;
# `RescaledCosmology.get_linP_Mpc_params` only applies these shifts.
#
# We compare, for many primordial changes:
#
# 1. **exact rescaling** (the new `RescaledCosmology.get_linP_Mpc_params`);
# 2. **polynomial fit of the rescaled power** (the generic
#    `BaseCosmology.get_linP_Mpc_params` applied to the same rescaled
#    cosmology): must agree with 1 to round-off;
# 3. **fresh CAMB** with the changed parameters: agrees up to CAMB numerical
#    precision (the transfer function is recomputed).

# %%
import time

import numpy as np

from lace.cosmo.base_cosmology import BaseCosmology
from lace.cosmo.cosmology import Cosmology
from lace.cosmo.rescale_cosmology import RescaledCosmology

fiducial = Cosmology()
fid_prim = fiducial.get_primordial_params()
print({key: fid_prim[key] for key in ("As", "ns", "nrun")})

# %% [markdown]
# ## Cosmologies, redshifts and pivots
#
# `kp_Mpc = 0.7 1/Mpc` is the emulator pivot; the velocity pivot
# `kp_kms = 0.009 s/km` at `z_star = 3` gives cup1d's $\Delta^2_\star$, $n_\star$,
# $\alpha_\star$.

# %%
rng = np.random.default_rng(42)
changes = [
    {"As": 1.2 * fid_prim["As"]},
    {"ns": fid_prim["ns"] - 0.03},
    {"nrun": 0.02},
    {"As": 0.8 * fid_prim["As"], "ns": fid_prim["ns"] + 0.03, "nrun": -0.02},
]
for _ in range(4):
    changes.append({
        "As": fid_prim["As"] * rng.uniform(0.8, 1.2),
        "ns": fid_prim["ns"] + rng.uniform(-0.04, 0.04),
        "nrun": rng.uniform(-0.03, 0.03),
    })
zs = [2.2, 3.0, 4.2]
kp_Mpc = 0.7
z_star, kp_kms = 3.0, 0.009
keys = ("Delta2_p", "n_p", "alpha_p")

# %% [markdown]
# ## Exact rescaling vs. polynomial fit of the rescaled power (same cosmology)

# %%
max_fit = {key: 0.0 for key in keys}
for params in changes:
    rescaled = RescaledCosmology(fiducial, params)
    for z in zs:
        exact = rescaled.get_linP_Mpc_params(z, kp_Mpc)
        fit = BaseCosmology.get_linP_Mpc_params(rescaled, z, kp_Mpc)
        for key in keys:
            diff = abs(exact[key] - fit[key])
            if key == "Delta2_p":
                diff /= abs(fit[key])
            max_fit[key] = max(max_fit[key], diff)
print("max |exact - fit|  (relative for Delta2_p):")
for key in keys:
    print(f"  {key:9s} {max_fit[key]:.2e}")

# %% [markdown]
# ## Exact rescaling vs. fresh CAMB
#
# One CAMB run per cosmology (a few seconds each).

# %%
print(f"{'change':55s} {'dDelta2_p/Delta2_p':>18s} {'dn_p':>10s} {'dalpha_p':>10s}"
      f" {'dDelta2_star/D':>15s} {'dn_star':>10s}")
for params in changes:
    rescaled = RescaledCosmology(fiducial, params)
    fresh = Cosmology(cosmo_params_dict=params)
    exact = rescaled.get_linP_Mpc_params(3.0, kp_Mpc)
    camb = fresh.get_linP_Mpc_params(3.0, kp_Mpc)
    exact_star = rescaled.get_linP_kms_params(z_star, kp_kms)
    camb_star = fresh.get_linP_kms_params(z_star, kp_kms)
    label = ", ".join(f"{k}={v:.4g}" for k, v in params.items())
    print(
        f"{label:55s} "
        f"{exact['Delta2_p'] / camb['Delta2_p'] - 1:18.2e} "
        f"{exact['n_p'] - camb['n_p']:10.2e} "
        f"{exact['alpha_p'] - camb['alpha_p']:10.2e} "
        f"{exact_star['Delta2_star'] / camb_star['Delta2_star'] - 1:15.2e} "
        f"{exact_star['n_star'] - camb_star['n_star']:10.2e}"
    )

# %% [markdown]
# For reference, the CAMB numerical noise itself: the fiducial cosmology
# recomputed from scratch against the original one.

# %%
again = Cosmology()
a = fiducial.get_linP_Mpc_params(3.0, kp_Mpc)
b = again.get_linP_Mpc_params(3.0, kp_Mpc)
print({key: (a[key] / b[key] - 1 if key == "Delta2_p" else a[key] - b[key]) for key in keys})

# %% [markdown]
# ## Speed

# %%
rescaled = RescaledCosmology(fiducial, changes[-1])
rescaled.get_linP_Mpc_params(3.0, kp_Mpc)  # fill the fiducial cache
n = 2000
start = time.perf_counter()
for _ in range(n):
    rescaled.get_linP_Mpc_params(3.0, kp_Mpc)
t_exact = (time.perf_counter() - start) / n
start = time.perf_counter()
for _ in range(200):
    BaseCosmology.get_linP_Mpc_params(rescaled, 3.0, kp_Mpc)
t_fit = (time.perf_counter() - start) / 200
print(f"exact rescaling: {1e6 * t_exact:.1f} us/call, polynomial fit: {1e6 * t_fit:.1f} us/call, "
      f"speed-up {t_fit / t_exact:.0f}x")
