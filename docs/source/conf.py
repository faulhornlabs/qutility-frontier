from __future__ import annotations

import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

project = "ScalableVolumetricBenchmark"
author = "ScalableVolumetricBenchmark contributors"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ✅ autosummary ON
autosummary_generate = True

# ✅ prevents imported/re-exported members from duplicating
autosummary_imported_members = False

# ✅ shorter class/function names in pages
add_module_names = False

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
