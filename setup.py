# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""NOTE: Put all metadata in pyproject.toml. Do not include complex logic in setup.py."""

import datetime
import os
import pathlib

import setuptools

# Logic for computing the development version number.
ROOT_DIR = pathlib.Path(__file__).parent
VERSION_FILE = ROOT_DIR / "VERSION"
version = VERSION_FILE.read_text().strip()

if os.environ.get("ONNX_SCRIPT_RELEASE") != "1":
    date = datetime.date.today().strftime("%Y%m%d")
    version = f"{version}.dev{date}"

# NOTE: Do not include other metadata in setup.py. Put it in pyproject.toml.
setuptools.setup(version=version)
