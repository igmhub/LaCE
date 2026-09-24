"""Validation of LaCE GP model bundles before unpickling their payloads."""
from __future__ import annotations

import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MODEL_SCHEMA_VERSION = 1


class ModelBundleError(RuntimeError):
    """A model bundle is absent, damaged, or incompatible."""


def manifest_path(folder: str | Path, model_label: str) -> Path:
    """Return the manifest for one full or leave-one-out model."""
    stem = Path(model_label).stem
    name = "manifest.json" if stem == "full" else f"manifest_{stem}.json"
    return Path(folder) / name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    result = {"python": ".".join(map(str, sys.version_info[:3]))}
    for package in ("numpy", "scipy", "scikit-learn"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def load_manifest(folder, emulator_label, label, normalization_path):
    """Validate a JSON bundle inventory before loading object arrays or pickles."""
    folder = Path(folder)
    path_manifest = manifest_path(folder, label)
    if not path_manifest.exists():
        return None
    try:
        manifest = json.loads(path_manifest.read_text(encoding="utf-8"))
    except OSError as error:
        raise ModelBundleError(f"Cannot read model manifest {path_manifest}: {error}") from error
    except json.JSONDecodeError as error:
        raise ModelBundleError(f"Invalid JSON model manifest {path_manifest}: {error}") from error
    if manifest.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ModelBundleError(f"Unsupported model manifest schema in {path_manifest}; retrain or migrate the bundle.")
    if manifest.get("emulator_label") != emulator_label or manifest.get("model_label") != label:
        raise ModelBundleError(f"Model manifest {path_manifest} does not identify {emulator_label!r}/{label!r}.")
    files = manifest.get("files")
    n_emulators = manifest.get("n_emulators")
    if not isinstance(files, dict) or not isinstance(n_emulators, int) or n_emulators < 1:
        raise ModelBundleError(f"Model manifest {path_manifest} has an invalid file inventory.")
    meta_name = "meta.npy" if label == "full.pkl" else f"meta_{Path(label).stem.removeprefix('drop_')}.npy"
    expected_names = {meta_name, *(f"n{index}_{label}" for index in range(n_emulators))}
    if set(files) != expected_names:
        raise ModelBundleError(f"Model manifest {path_manifest} has an incomplete or unexpected file inventory.")
    for name, expected in files.items():
        item = folder / name
        if not item.is_file():
            raise ModelBundleError(f"Required model file is missing: {item}. Configure model_path/data_path or restore a complete trusted bundle.")
        if not isinstance(expected, str) or _sha256(item) != expected:
            raise ModelBundleError(f"Checksum mismatch for {item}; obtain a complete trusted model bundle.")
    normalization = manifest.get("normalization")
    path_normalization = Path(normalization_path)
    if not isinstance(normalization, dict) or not isinstance(normalization.get("sha256"), str):
        raise ModelBundleError(f"Model manifest {path_manifest} has no valid normalization checksum.")
    if not path_normalization.is_file():
        raise ModelBundleError(f"Required normalization file is missing: {path_normalization}.")
    if _sha256(path_normalization) != normalization["sha256"]:
        raise ModelBundleError(f"Checksum mismatch for normalization file {path_normalization}.")
    dependencies = manifest.get("dependencies", {})
    if not manifest.get("legacy", False) and isinstance(dependencies, dict):
        wanted = dependencies.get("scikit-learn")
        installed = runtime_versions()["scikit-learn"]
        if wanted and wanted != installed:
            raise ModelBundleError(f"Model requires scikit-learn {wanted}, but {installed} is installed. Use the pinned environment or retrain/migrate and validate the model.")
    return manifest


def write_manifest(folder, emulator_label, label, files, normalization_path, drop_sim, provenance=None):
    folder = Path(folder)
    import lace
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "emulator_label": emulator_label,
        "model_label": label,
        "excluded_simulation": drop_sim,
        "training_provenance": provenance or {"status": "not recorded"},
        "lace_version": getattr(lace, "__version__", "unknown"),
        "dependencies": runtime_versions(),
        "n_emulators": len(files) - 1,
        "files": {item.name: _sha256(item) for item in files},
        "normalization": {"filename": Path(normalization_path).name, "sha256": _sha256(Path(normalization_path))},
    }
    target = manifest_path(folder, label)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
