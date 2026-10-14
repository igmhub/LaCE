"""Persistent local paths for external LaCE data."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

DEFAULT_NYX_PATH = Path(
    "/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files"
)


def _path_config(config_path: str | Path | None) -> Path:
    """Normalize an explicit configuration path or choose the user default.

    Parameters
    ----------
    config_path : str or pathlib.Path, optional
        Explicit TOML configuration file.

    Returns
    -------
    pathlib.Path
        Expanded explicit path or :func:`get_paths_config_path`.
    """
    return Path(config_path).expanduser() if config_path is not None else get_paths_config_path()


def _configured_path(key: str, default: Path, config_path: str | Path | None) -> Path:
    """Read one path entry from persistent TOML configuration.

    Parameters
    ----------
    key : str
        Name below the TOML ``[paths]`` table.
    default : pathlib.Path
        Fallback when file or key is absent.
    config_path : str or pathlib.Path, optional
        Explicit configuration file.

    Returns
    -------
    pathlib.Path
        Expanded configured or default path.

    Raises
    ------
    ValueError
        If TOML is malformed or a configured path is blank.
    TypeError
        If a configured path is not a string.
    """
    path_config = _path_config(config_path)
    if not path_config.exists():
        return default
    try:
        with path_config.open("rb") as stream:
            config = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Could not read LaCE paths configuration at {path_config}") from error
    configured_path = config.get("paths", {}).get(key)
    if configured_path is None:
        return default
    if not isinstance(configured_path, str):
        raise TypeError(f"paths.{key} in {path_config} must be a string")
    if not configured_path.strip():
        raise ValueError(f"paths.{key} in {path_config} must not be blank")
    return Path(configured_path).expanduser()


def _set_configured_path(key: str, path: Path, config_path: str | Path | None) -> Path:
    """Update one ``[paths]`` value without discarding other TOML settings.

    Parameters
    ----------
    key : str
        TOML ``[paths]`` entry name.
    path : pathlib.Path
        Expanded path persisted as a JSON-quoted TOML string.
    config_path : str or pathlib.Path, optional
        Explicit configuration file.

    Returns
    -------
    pathlib.Path
        Persisted path.
    """
    path_config = _path_config(config_path)
    path_config.parent.mkdir(parents=True, exist_ok=True)
    text = path_config.read_text(encoding="utf-8") if path_config.exists() else ""
    value_line = f"{key} = {json.dumps(str(path))}"
    section = re.search(r"(?m)^\[paths\][ \t]*(?:#.*)?$", text)
    if section is None:
        separator = "" if not text or text.endswith("\n") else "\n"
        text = f"{text}{separator}[paths]\n{value_line}\n"
    else:
        next_section = re.search(r"(?m)^\[[^]]+\][ \t]*(?:#.*)?$", text[section.end():])
        end = section.end() + (next_section.start() if next_section else len(text[section.end():]))
        body = text[section.end():end]
        if body and not body.startswith("\n"):
            body = "\n" + body
        line = re.compile(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=.*$")
        if line.search(body):
            body = line.sub(value_line, body, count=1)
        else:
            if body and not body.endswith("\n"):
                body += "\n"
            body += value_line + "\n"
        text = text[:section.end()] + body + text[end:]
    path_config.write_text(text, encoding="utf-8")
    return path


def get_path_repo() -> Path:
    """Return the LaCE repository root inferred from the installed package.

    This is useful for developer scripts and notebooks that operate on the
    repository's ``data`` directory.  It works naturally for an editable
    installation, without requiring a ``LACE_REPO`` environment variable.
    """

    import lace

    return Path(lace.__path__[0]).parent


def get_paths_config_path() -> Path:
    """Return the user-level TOML file storing local LaCE paths.

    Returns
    -------
    pathlib.Path
        ``~/.config/lace/paths.toml``.
    """

    return Path.home() / ".config" / "lace" / "paths.toml"


def get_nyx_path(
    nyx_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Return the configured Nyx data directory.

    Parameters
    ----------
    nyx_path : str or pathlib.Path, optional
        Explicit directory taking priority over persistent configuration.
    config_path : str or pathlib.Path, optional
        Explicit configuration file.

    Returns
    -------
    pathlib.Path
        Expanded Nyx directory; defaults to :data:`DEFAULT_NYX_PATH`.

    An explicitly supplied path takes priority. Otherwise, use ``nyx_path``
    from ``~/.config/lace/paths.toml`` when available, falling back to the
    NERSC DESI location in :data:`DEFAULT_NYX_PATH`.
    """

    if nyx_path is not None:
        return Path(nyx_path).expanduser()

    return _configured_path("nyx_path", DEFAULT_NYX_PATH, config_path)

def set_nyx_path(nyx_path: str | Path, config_path: str | Path | None = None) -> Path:
    """Persist a local Nyx data directory and return its normalized path.

    Parameters
    ----------
    nyx_path : str or pathlib.Path
        Nonblank Nyx directory.
    config_path : str or pathlib.Path, optional
        Explicit configuration file.

    Returns
    -------
    pathlib.Path
        Persisted expanded Nyx path.

    Raises
    ------
    ValueError
        If a string path is blank.

    Notes
    -----
    This writes ``~/.config/lace/paths.toml``. It is intended for machines
    where the Nyx archive is not available at the default NERSC location.
    """

    path = Path(nyx_path).expanduser()
    if isinstance(nyx_path, str) and not nyx_path.strip():
        raise ValueError("nyx_path must not be blank")
    return _set_configured_path("nyx_path", path, config_path)


def get_data_path(data_path: str | Path | None = None, config_path: str | Path | None = None) -> Path:
    """Return the root containing external LaCE runtime assets.

    Parameters
    ----------
    data_path : str or pathlib.Path, optional
        Explicit runtime-asset root.
    config_path : str or pathlib.Path, optional
        Explicit configuration file.

    Returns
    -------
    pathlib.Path
        Expanded configured or checkout-local data root.

    Raises
    ------
    ValueError
        If an explicit string path is blank.

    Notes
    -----
    Explicit arguments take precedence over ``paths.data_path`` and the
    checkout's ``data`` directory. Installed wheels normally configure this
    location because simulation and model assets are deliberately external.
    """
    if data_path is not None:
        if isinstance(data_path, str) and not data_path.strip():
            raise ValueError("data_path must not be blank")
        return Path(data_path).expanduser()
    return _configured_path("data_path", get_path_repo() / "data", config_path)


def set_data_path(data_path: str | Path, config_path: str | Path | None = None) -> Path:
    """Persist the root containing GP models, normalizations, and Gadget data.

    Parameters
    ----------
    data_path : str or pathlib.Path
        Nonblank runtime-asset root.
    config_path : str or pathlib.Path, optional
        Explicit configuration file.

    Returns
    -------
    pathlib.Path
        Persisted expanded data root.

    Raises
    ------
    ValueError
        If a string path is blank.
    """
    if isinstance(data_path, str) and not data_path.strip():
        raise ValueError("data_path must not be blank")
    return _set_configured_path("data_path", Path(data_path).expanduser(), config_path)
