# SPDX-License-Identifier: Apache-2.0


# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import sys
import onnxscript


# -- Project information -----------------------------------------------------

project = 'onnx-script'
copyright = '2022, onnx'
author = 'onnx'
version = onnxscript.__version__
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
html_logo = "logo_main.png"

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
    'examples_dirs': ['examples', 'tutorial'],
    'gallery_dirs': ['auto_examples', 'auto_tutorial'],
    'capture_repr': ('_repr_html_', '__repr__'),
    'ignore_repr_types': r'matplotlib.text|matplotlib.axes',
    'binder': {
        'org': 'onnx',
        'repo': 'onnx.ai/onnxscript/',
        'notebooks_dir': 'auto_examples',
        'binderhub_url': 'https://mybinder.org',
        'branch': 'master',
        'dependencies': './requirements.txt'
    },
}
