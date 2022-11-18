# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys

from .backend.onnx_export import export2python as proto2python
from .main import export_onnx_lib, graph, script
from .onnx_types import (
    BFLOAT16,
    BOOL,
    COMPLEX64,
    COMPLEX128,
    DOUBLE,
    FLOAT,
    FLOAT16,
    INT8,
    INT16,
    INT32,
    INT64,
    STRING,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
)
from .utils import external_tensor, proto2text
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
    __version__ = None  # type: ignore[assignment]


__all__ = [
    "script",
    "export_onnx_lib",
    "OnnxFunction",
    "proto2python",
    "proto2text",
    "external_tensor",
    "graph",
]
