# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'LaCE'
copyright = '2024, Andreu Font-Ribera, Laura Cabayol-Garcia, Jonas Chaves-Montero, Christian Pedersen'
author = 'Andreu Font-Ribera, Laura Cabayol-Garcia, Jonas Chaves-Montero, Christian Pedersen'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}

# Ignore specific warnings
suppress_warnings = [
    'ref.ref',  # Ignores undefined references
    'ref.python',  # Ignores Python cross-references
]

templates_path = ['_templates']
exclude_patterns = ['.ipynb_checkpoints']

language = 'En'
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

