# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.rewriter import _ir_utils, pattern

def _rotate_half_pattern(op, x, start1, end1, start2, end2):
    # Slice(input, starts, ends, axes, steps)
    x1 = op.Slice(x, start1, end1, [3], [1])
    x2 = op.Slice(x, start2, end2, [3], [1])
    minus_x2 = op.Neg(x2)
    rotated_x = op.Concat(minus_x2, x1, axis=-1)
    return rotated_x

class RotaryEmbeddingFusion(pattern.RewriteRuleClassBase):

    def pattern(self, op, x, cos, sin, start1, end1, start2, end2):
        return x * cos + _rotate_half_pattern(op, x, start1, end1, start2, end2) * sin

    def check(self, op, x, start1, end1, start2, end2, **_):
        # x needs to be a 4D tensor with known last dimension size (== head_size)
        if x is None or x.shape is None or len(x.shape) != 4:
            return False
        head_size = x.shape[3]
        if not isinstance(head_size, int):
            return False
        half_head_size = head_size // 2

        # Check that x is being split into two equal halves of size half_head_size
        return (
            _ir_utils.is_singleton_value(start1, 0) and
            _ir_utils.is_singleton_value(end1, half_head_size) and
            _ir_utils.is_singleton_value(start2, half_head_size) and
            _ir_utils.is_singleton_value(end2, lambda x: x >= head_size)
        )

    def rewrite(self, op, x, cos, sin, **_):
        return op.RotaryEmbedding(x, cos, sin, interleaved=0, _domain="ai.onnxruntime.fusion")



_rule = RotaryEmbeddingFusion.rule()

rotary_embedding_rules = pattern.RewriteRuleSet([_rule])

def fuse_rotary_embedding(model: ir.Model) -> None:
    count = rotary_embedding_rules.apply_to_model(model)
    print(f"Rotary Embedding count: {count}")
