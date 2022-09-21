# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from .main import script, export_onnx_lib, OnnxFunction
from .backend.onnx_export import export2python as proto2python

if sys.version_info[0:2] >= (3, 8):
    import importlib.metadata as importlib_metadata
else:
    # TODO: Remove this when Python 3.7 is deprecated
    import importlib_metadata

# TODO: should we algin the folder name with pacakge name? 
# It's onnxscript and onnx-script now. That way, we can use __package__ here.
__version__ = importlib_metadata.version(__package__)

__all__ = [script, export_onnx_lib, OnnxFunction, proto2python]
