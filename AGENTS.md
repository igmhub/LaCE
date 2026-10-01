# AGENTS.md — LaCE

## Purpose and active layout

LaCE owns cosmological calculations, simulation archives, and P1D GP emulators. ForestFlow and cup1d are downstream consumers; keep this package independent of both.

- Cosmology: `lace/cosmo/`; archives: `lace/archive/`; production emulator factory: `lace/emulator/emulator_manager.py`; GP implementation: `lace/emulator/gp_emulator_multi.py`.
- External paths: `lace/configuration/`; naming contracts: `lace/conventions.py`; model provenance: `lace/emulator/model_manifest.py`.
- Supported factory labels currently include `CH24_mpgcen_gpr` and `CH24_nyxcen_gpr`. Other labels in old code or low-level classes do not automatically define a supported production interface.
- Tutorials: `notebooks/tutorials/`; scientific regressions: `tests/`. `lace/old_code/` is deprecated; `notebooks/cosmology/wip/` is exploratory.

## Working rules

- Inspect `git status`, the current branch, and applicable nested instructions before editing. The maintained development branch is `vectorize`; do not switch branches or discard user changes automatically.
- Read the relevant implementation, tests, and `docs/workflow*` before changing a public interface. Follow active imports rather than assuming every notebook defines supported behavior.
- Make focused changes. Preserve scientific defaults, parameter ordering, serialization, and scalar/batch behavior unless the task explicitly changes them. Document intentional numerical changes and their validation.
- Do not edit `old_code/`, notebook `old/`, `wip/`, or developer experiments by default. Historical/paper modules are not necessarily deprecated: check callers first. Never copy an obsolete API back into the active package without checking it.
- Do not regenerate simulation archives, model weights, covariance products, chains, or publication outputs as part of a routine code change. Use configured external assets and report missing prerequisites. Never replace a scientific regression reference solely to make a test pass.
- Use Python >=3.12 and editable installs. Install sibling IGMHub repositories explicitly from compatible revisions, rather than relying on the PyPI name `lace`. Record the three commit SHAs for cross-package validation.
- Distinguish fast unit checks from model/data-dependent regression tests. A skipped or unavailable regression is not a pass. Prefer a small test of the failing scientific invariant over a test that merely reproduces the implementation.
- Maintain Jupytext `.py` notebook sources; sync only the affected pair with `jupytext --sync path/to/notebook.py`. Avoid generating every notebook for an unrelated change.
- Versions are derived from Git via setuptools-scm; do not hand-edit generated `_version.py` files. Update API docstrings and relevant documentation when behavior changes.
- Report what changed, commands actually run, missing assets/dependencies, and any numerical or scientific limitations.

## Shared scientific contracts

- Consult each package's `conventions.py`. Canonical public names include `k_iMpc`, `k_ikms`, `P1D_Mpc`, `P1D_kms`, `P3D_Mpc`, and `dkms_diMpc`. Existing serialized data and APIs retain legacy spellings; translate at explicit boundaries rather than silently renaming stored products.
- With `M = H(z)/(1+z)` in km/s/Mpc: `k_iMpc = M * k_ikms`, `P1D_kms = M * P1D_Mpc`, and P1D covariance gains two factors of M. P3D has volume units (Mpc^3); do not apply a P1D Jacobian to it.
- Preserve the distinction between comoving Mpc and Mpc/h, thermal broadening length `sigT_Mpc`, and inverse pressure smoothing scale `kF_Mpc`.
- Linear-power defaults distinguish baryon+CDM (`bc`) from total matter (`bcnu`). Check species, pivot, redshift, primordial running convention, and growth convention before comparing predictions.
- Primordial rescaling is only valid when all transfer-function/background parameters are unchanged. Changes in neutrino mass, effective relativistic species, dark energy, curvature, or densities require an appropriate fresh cosmology calculation.
- Treat scalar, redshift, k, batch, and stochastic-sample axes explicitly. Use unequal axis lengths in tests to expose accidental broadcasting; preserve ragged observational k grids.
- Covariance must preserve data ordering and selected cross-bin correlations. Validate symmetry, finite entries, and positive definiteness; do not hide invalid matrices with absolute determinants or arbitrary regularization.

## LaCE-specific safeguards

- Use `Cosmology`/`BaseCosmology` as the shared cosmology interface. Validate `RescaledCosmology` against fresh CAMB for supported primordial changes, and explicitly reject incompatible transfer-function/background changes, including `nnu`.
- Archive averaging, optical-depth rescaling, held-out simulations/redshifts, and central-simulation inclusion alter training data scientifically. Preserve them and check all selection paths; validation splits must not leak held-out data.
- Check mean-flux normalization, smoothing, interpolation/extrapolation bounds, training-domain membership, and parameter order before modifying GP predictions.
- Require public canonical aliases to preserve argument meaning and return contracts. Do not promise GP covariance if the production implementation does not provide it.
- Simulation suites and L1O bundles are externally configured. Use `get_data_path`/`get_nyx_path` and explicit overrides; do not add new hard-coded NERSC paths. Keep manifest validation and trusted-model provenance intact.
- Test cache invalidation when alternating cosmologies or inputs. Repeated calls and permutations must give the same physical prediction.

## Validation commands

```bash
python -m pip install -e ".[test]"
pytest -q
```

Start with the relevant `tests/test_configuration.py`, `test_conventions.py`, `test_cosmology.py`, `test_rescaled_cosmology.py`, `test_gadget_archive.py`, `test_emulator.py`, or `test_model_manifest.py`. Some archive/emulator tests need configured model/data assets; inspect failures and skips rather than substituting models. For changed cosmology/emulator APIs also validate downstream ForestFlow/cup1d at compatible SHAs. For documentation changes install `.[docs]` and run `make docs`.
