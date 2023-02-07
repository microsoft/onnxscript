# Configuration file for the Sphinx documentation builder.
# To run the documentation: python -m sphinx docs dist/html

import os
import re
import sys

import sphinx_gallery.sorting

import onnxscript

# -- Project information -----------------------------------------------------

project = "onnx-script"
copyright = "2023, onnx"
author = "onnx"
version = onnxscript.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgmath",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
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
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "onnxruntime": ("https://onnxruntime.ai/docs/api/python/", None),
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
