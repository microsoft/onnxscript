# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ONNX Function definitions for proposed linear attention operators.

These functions define reference implementations using standard ONNX ops
for the operators proposed in https://github.com/onnx/onnx/issues/7689.
They serve as:

1. **Semantic specifications** — precise mathematical definitions of each op
2. **Fallback implementations** — backends that don't have native kernels
   can expand the function body and execute via standard ops

Each function returns an ``ir.Function`` that can be attached to an
``ir.Model`` or used as a rewrite target.
"""

from __future__ import annotations

from mobius.functions.causal_conv import (
    causal_conv1d_with_state,
)
from mobius.functions.linear_attention import (
    linear_attention,
)

__all__ = [
    "causal_conv1d_with_state",
    "linear_attention",
]
