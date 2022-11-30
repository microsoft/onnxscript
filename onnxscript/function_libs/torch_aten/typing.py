# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from onnxscript import DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, TensorType

FloatType = FLOAT16 | FLOAT | DOUBLE
IntType = INT16 | INT32 | INT64
