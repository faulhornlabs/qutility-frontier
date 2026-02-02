from __future__ import annotations

import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

project = "qutility-frontier"
author = "Portik Attila (Qutility @ Faulhorn Labs)"
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

autosummary_generate = True

autosummary_imported_members = False

add_module_names = False

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

html_theme = "sphinx_rtd_theme"
