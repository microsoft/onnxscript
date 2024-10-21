# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np
import onnxscript.ir as ir
from onnxscript.rewriter import pattern

def rotate_half_pattern(op, x, start1, end1, start2, end2):
    # Slice(input, starts, ends, axes, steps)
    x1 = op.Slice(x, start1, end1, [3], [1])
    x2 = op.Slice(x, start2, end2, [3], [1])
    minus_x2 = op.Neg(x2)
    rotated_x = op.Concat(minus_x2, x1, axis=-1)
    return rotated_x

def rotate_half(op, x, start1, end1, start2, end2):
    # TODO: check if start1, end1, start2, end2 are valid
    return op.RotateHalf(x, _domain="local")

rule = pattern.RewriteRule(rotate_half_pattern, rotate_half)