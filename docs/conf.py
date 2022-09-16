# Configuration file for the Sphinx documentation builder.
# To run the documentation: python -m sphinx docs dist/html

import sys
import re
import os
import sphinx_gallery.sorting
import sys

# single-source ONNXScript version
# python version decides different lib
# if sys.version_info[:2] >= (3, 8):
from importlib import metadata
# else:
#     import importlib_metadata as metadata
__version__ = metadata.version(__package__)
del metadata

# -- Project information -----------------------------------------------------

project = 'onnx-script'
copyright = '2022, onnx'
author = 'onnx'
version = __version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    "sphinx.ext.autodoc",
    'sphinx.ext.githubpages',
    "sphinx_gallery.gen_gallery",
    'sphinx.ext.autodoc',
    'sphinx.ext.graphviz',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
source_suffix = ['.rst']

master_doc = 'index'
language = "en"
exclude_patterns = []
pygments_style = 'default'

# -- Options for HTML output -------------------------------------------------

html_static_path = ['_static']
html_theme = "pydata_sphinx_theme"
html_theme_path = ['_static']
html_logo = "_static/logo_main.png"

# -- Options for graphviz ----------------------------------------------------

graphviz_output_format = "svg"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for Sphinx Gallery ----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'onnxruntime': ('https://onnxruntime.ai/docs/api/python/', None),
}

sphinx_gallery_conf = {
    'examples_dirs': ['examples'],
    'gallery_dirs': ['auto_examples'],
    'capture_repr': ('_repr_html_', '__repr__'),
    'ignore_repr_types': r'matplotlib.text|matplotlib.axes',
    'filename_pattern': re.escape(os.sep) + '[0-9]*_?plot_',
    'within_subsection_order': sphinx_gallery.sorting.FileNameSortKey,
}
