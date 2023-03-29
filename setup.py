# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Setup script for the onnxscript package.

NOTE: Put all metadata in pyproject.toml.
"""
import os
import pathlib
import setuptools
import sys

from datetime import date

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
package_name = "onnx-script"

with open(os.path.join(TOP_DIR, "VERSION_NUMBER")) as version_file:
    VERSION_NUMBER = version_file.read().strip()
    if "--weekly_build" in sys.argv:
        today_number = date.today().strftime("%Y%m%d")
        VERSION_NUMBER += ".dev" + today_number
        PACKAGE_NAME = "onnx_script_weekly"
        sys.argv.remove("--weekly_build")

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION_NUMBER,
    description="ONNX Script",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=packages,
    license="MIT License",
    url="https://github.com/microsoft/onnx-script",
)
