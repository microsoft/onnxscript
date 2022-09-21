# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import os
from distutils.core import setup
from setuptools import find_packages

this = os.path.dirname(__file__)

packages = find_packages()
assert packages

README = os.path.join(os.getcwd(), "README.md")
with open(README, "r") as f:
    long_description = f.read()
    start_pos = long_description.find('## Contributing')
    if start_pos >= 0:
        long_description = long_description[:start_pos]

setup(
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/onnx/onnx-script',
    packages=packages,
    include_package_data=True,
    package_data={"onnx-script": ["py.typed"], "onnx": ["py.typed"],},
)
