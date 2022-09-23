# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys

from .backend.onnx_export import export2python as proto2python
from .utils import external_tensor, proto2text

from .main import export_onnx_lib, script
from .values import OnnxFunction

if sys.version_info[0:2] >= (3, 8):
    import importlib.metadata as importlib_metadata
else:
    # TODO: Remove this when Python 3.7 is deprecated
    import importlib_metadata

try:
    # TODO: should we algin the folder name with package name?
    # It's onnxscript and onnx-script now. That way, we can use __package__ here.
    __version__ = importlib_metadata.version("onnx-script")
except importlib_metadata.PackageNotFoundError:
    __version__ = None


__all__ = ["script", "export_onnx_lib", "OnnxFunction", "proto2python", "proto2text", "external_tensor"]
