# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

# project_root = docs/source/../../
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

project = "ScalableVolumetricBenchmark"
copyright = "2025, Attila Portik"
author = "Attila Portik"
release = "0.0.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # pull docs from docstrings
    "sphinx.ext.napoleon",  # support Google/NumPy docstring styles
    "sphinx.ext.viewcode",  # add links to highlighted source code
    "sphinx.ext.autosummary",  # (optional) summary tables
    "sphinx.ext.mathjax",  # for LaTeX-style math like in the screenshot
]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "titles_only": False,
}
html_static_path = ["_static"]
