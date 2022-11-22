# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys

from .backend.onnx_export import export2python as proto2python
from .main import export_onnx_lib, graph, script

# isort: off
from .onnx_opset import (
    opset1,
    opset2,
    opset3,
    opset4,
    opset5,
    opset6,
    opset7,
    opset8,
    opset9,
    opset10,
    opset11,
    opset12,
    opset13,
    opset14,
    opset15,
    opset16,
    opset17,
    opset18,
    default_opset,
    onnxml1,
    onnxml2,
    onnxml3,
)

from .onnx_types import (
    BFLOAT16,
    FLOAT16,
    FLOAT,
    DOUBLE,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    BOOL,
    STRING,
    COMPLEX64,
    COMPLEX128,
)

# isort: on
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
    "BFLOAT16",
    "FLOAT16",
    "FLOAT",
    "DOUBLE",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "BOOL",
    "STRING",
    "COMPLEX64",
    "COMPLEX128",
    "opset1",
    "opset2",
    "opset3",
    "opset4",
    "opset5",
    "opset6",
    "opset7",
    "opset8",
    "opset9",
    "opset10",
    "opset11",
    "opset12",
    "opset13",
    "opset14",
    "opset15",
    "opset16",
    "opset17",
    "opset18",
    "default_opset",
    "onnxml1",
    "onnxml2",
    "onnxml3",
]
