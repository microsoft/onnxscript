# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""NOTE: Put all metadata in pyproject.toml. Do not include complex logic in setup.py."""

import pathlib

import setuptools

# Logic for computing the development version number.
ROOT_DIR = pathlib.Path(__file__).parent
VERSION_FILE = ROOT_DIR / "VERSION"
version = VERSION_FILE.read_text().strip()

# NOTE: Do not include other metadata in setup.py. Put it in pyproject.toml.
setuptools.setup(version=version)
