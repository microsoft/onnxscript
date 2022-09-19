# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys

from .backend.onnx_export import export2python as proto2python
from .main import OnnxFunction, export_onnx_lib, script

if sys.version_info[0:2] >= (3, 8):
    import importlib.metadata as importlib_metadata
else:
    # In Python 3.7, importlib.metadata is not available
    import importlib_metadata

try:
    __version__ = importlib_metadata.version(__package__)
except Exception:  # nosec: allow bare except
    # If the package is not installed or the version lookup fails
    __version__ = None

__all__ = [script, export_onnx_lib, OnnxFunction, proto2python]
