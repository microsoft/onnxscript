# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Configuration file for the Sphinx documentation builder.

To run the documentation: python -m sphinx docs dist/html
"""

import os
import re
import sys

import sphinx_gallery.sorting

import onnxscript

# -- Project information -----------------------------------------------------

project = "onnxscript"
copyright = "Microsoft. All rights reserved."
author = "onnx"
version = onnxscript.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_exec_code",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "attrs_block",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]

master_doc = "index"
language = "en"
exclude_patterns = []
pygments_style = "default"

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]
html_theme = "furo"
html_theme_path = ["_static"]
html_theme_options = {
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    "sidebar_hide_name": True,
}
html_css_files = ["css/custom.css"]

# -- Options for graphviz ----------------------------------------------------

graphviz_output_format = "svg"

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "onnx": ("https://onnx.ai/onnx/", None),
    "onnx_ir": ("https://onnx.ai/ir-py/", None),
    "onnxruntime": ("https://onnxruntime.ai/docs/api/python/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/main/", None),
}

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    "capture_repr": ("_repr_html_", "__repr__"),
    "ignore_repr_types": r"matplotlib.text|matplotlib.axes",
    "filename_pattern": f"{re.escape(os.sep)}[0-9]*_?plot_",
    "within_subsection_order": sphinx_gallery.sorting.FileNameSortKey,
}
