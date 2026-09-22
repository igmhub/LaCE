"""Persistent local paths for external LaCE data."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path


DEFAULT_NYX_PATH = Path(
    "/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files"
)


def get_paths_config_path() -> Path:
    """Return the user-level file that stores local LaCE data paths."""

    return Path.home() / ".config" / "lace" / "paths.toml"


def get_nyx_path(nyx_path: str | Path | None = None) -> Path:
    """Return the configured Nyx data directory.

    An explicitly supplied path takes priority. Otherwise, use ``nyx_path``
    from ``~/.config/lace/paths.toml`` when available, falling back to the
    NERSC DESI location in :data:`DEFAULT_NYX_PATH`.
    """

    if nyx_path is not None:
        return Path(nyx_path).expanduser()

    config_path = get_paths_config_path()
    if not config_path.exists():
        return DEFAULT_NYX_PATH

    try:
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
        configured_path = config.get("paths", {}).get("nyx_path")
    except tomllib.TOMLDecodeError as error:
        raise ValueError(
            f"Could not read LaCE paths configuration at {config_path}"
        ) from error

    if configured_path is None:
        return DEFAULT_NYX_PATH
    if not isinstance(configured_path, str):
        raise TypeError(
            f"paths.nyx_path in {config_path} must be a string"
        )
    return Path(configured_path).expanduser()


def set_nyx_path(nyx_path: str | Path) -> Path:
    """Persist a local Nyx data directory and return its normalized path.

    This writes ``~/.config/lace/paths.toml``. It is intended for machines
    where the Nyx archive is not available at the default NERSC location.
    """

    path = Path(nyx_path).expanduser()
    config_path = get_paths_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "[paths]\nnyx_path = " + json.dumps(str(path)) + "\n",
        encoding="utf-8",
    )
    return path
