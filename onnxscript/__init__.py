# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

__all__ = [
    "script",
    "graph",
    "ir",
    "optimizer",
    "rewriter",
    "export_onnx_lib",
    "OnnxFunction",
    "TracedOnnxFunction",
    "proto2python",
    "external_tensor",
    "BFLOAT16",
    "FLOAT16",
    "FLOAT8E4M3FN",
    "FLOAT8E4M3FNUZ",
    "FLOAT8E5M2",
    "FLOAT8E5M2FNUZ",
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
    "opset19",
    "opset20",
    "opset_ai_onnx_ml1",
    "opset_ai_onnx_ml2",
    "opset_ai_onnx_ml3",
    "opset_ai_onnx_ml4",
    "DEBUG",
]

import importlib.metadata

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
    opset19,
    opset20,
    opset_ai_onnx_ml1,
    opset_ai_onnx_ml2,
    opset_ai_onnx_ml3,
    opset_ai_onnx_ml4,
)

from .onnx_types import (
    BFLOAT16,
    FLOAT16,
    FLOAT8E4M3FN,
    FLOAT8E4M3FNUZ,
    FLOAT8E5M2,
    FLOAT8E5M2FNUZ,
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

from . import ir, optimizer, rewriter
from ._internal.utils import external_tensor
from .values import OnnxFunction, TracedOnnxFunction

# Set DEBUG to True to enable additional debug checks
DEBUG = False

try:  # noqa: SIM105
    __version__ = importlib.metadata.version("onnxscript")
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    pass
