# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import os

import setuptools

this = os.path.dirname(__file__)

packages = setuptools.find_packages()
assert packages

README = os.path.join(os.getcwd(), "README.md")
with open(README, encoding="utf-8") as f:
    long_description = f.read()
    start_pos = long_description.find("## Contributing")
    if start_pos >= 0:
        long_description = long_description[:start_pos]

setuptools.setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/onnx/onnx-script",
    packages=packages,
    include_package_data=True,
    package_data={
        "onnx-script": ["py.typed"],
        "onnx": ["py.typed"],
    },
)
