"""Sphinx configuration for the LaCE documentation."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).parent / "_build/matplotlib")
)

project = "LaCE"
author = "LaCE developers"
try:
    release = importlib.metadata.version("lace")
except importlib.metadata.PackageNotFoundError:
    release = "development"

extensions = [
    "sphinx.ext.graphviz",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
if importlib.util.find_spec("numpydoc") is not None:
    extensions.insert(0, "numpydoc")

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
if "numpydoc" in extensions:
    numpydoc_show_class_members = False

# API discovery should remain available when building documentation without
# installing optional scientific backends or external simulation data.
autodoc_mock_imports = [
    "camb",
    "corner",
    "configobj",
    "matplotlib",
    "sklearn",
    "strenum",
    "torch",
]

html_theme = (
    "pydata_sphinx_theme"
    if importlib.util.find_spec("pydata_sphinx_theme") is not None
    else "alabaster"
)
html_title = "LaCE"
html_theme_options = (
    {"show_toc_level": 2, "navigation_with_keys": True}
    if html_theme == "pydata_sphinx_theme"
    else {}
)

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
intersphinx_mapping = {}
if os.environ.get("LACE_DOCS_INTERSPHINX") == "1":
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
    }
