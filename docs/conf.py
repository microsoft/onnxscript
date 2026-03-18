"""Configuration file for the Sphinx documentation builder.

To build the documentation: python -m sphinx docs docs/_build/html
"""

from __future__ import annotations

import sys

# -- Project information -----------------------------------------------------

project = "mobius"
copyright = "2026, ONNX Project Contributors"
author = "ONNX Project Contributors"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]

master_doc = "index"
language = "en"
exclude_patterns = ["_build"]
pygments_style = "default"

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = "mobius"

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "onnx": ("https://onnx.ai/onnx/", None),
}
