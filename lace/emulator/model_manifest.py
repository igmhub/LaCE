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


def load_manifest(folder: str | Path, emulator_label: str, label: str) -> dict | None:
    """Validate a safe JSON manifest before any NumPy object array or pickle load.

    A missing manifest is an explicitly supported legacy bundle. Its provenance
    cannot be recovered and is therefore never inferred.
    """
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ModelBundleError(f"Cannot read model manifest {manifest_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ModelBundleError(f"Invalid JSON model manifest {manifest_path}: {error}") from error
    if manifest.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ModelBundleError(f"Unsupported model manifest schema in {manifest_path}; retrain or migrate the bundle.")
    if manifest.get("emulator_label") != emulator_label or manifest.get("model_label") != label:
        raise ModelBundleError(f"Model manifest {manifest_path} does not identify {emulator_label!r}/{label!r}.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ModelBundleError(f"Model manifest {manifest_path} has no file inventory.")
    for name, expected in files.items():
        path = folder / name
        if not path.is_file():
            raise ModelBundleError(f"Required model file is missing: {path}. Configure model_path/data_path or restore a complete trusted bundle.")
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise ModelBundleError(f"Checksum mismatch for {path}; obtain a complete trusted model bundle.")
    dependencies = manifest.get("dependencies", {})
    if not manifest.get("legacy", False) and isinstance(dependencies, dict):
        installed = runtime_versions()
        wanted = dependencies.get("scikit-learn")
        if wanted and wanted != installed["scikit-learn"]:
            raise ModelBundleError(f"Model requires scikit-learn {wanted}, but {installed['scikit-learn']} is installed. Use the pinned environment or retrain/migrate and validate the model.")
    return manifest


def write_manifest(folder: str | Path, emulator_label: str, label: str, files: list[Path], drop_sim: str | None, provenance: dict | None = None) -> Path:
    folder = Path(folder)
    import lace
    lace_version = getattr(lace, "__version__", "unknown")
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "emulator_label": emulator_label,
        "model_label": label,
        "excluded_simulation": drop_sim,
        "training_provenance": provenance or {"status": "not recorded"},
        "lace_version": lace_version,
        "dependencies": runtime_versions(),
        "files": {path.name: _sha256(path) for path in files},
    }
    target = folder / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
