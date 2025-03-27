# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnxscript.ir as ir
from onnxscript.rewriter import _ir_utils, pattern

# Add first version of the RotaryEmbeddingFusion rule. This considers only one simple pattern
# for full rotation without interleaving.
# TODO(rama): Add pattern variations to handle other cases (interleaved, as well as partial rotation).

# Note: This targets the new op being proposed to ONNX. This version does not exist in ORT yet.
# so it can't be tested by running against ORT. See cos_sin_cache.py for a transformation that
# rewrites the pattern into one that can be run against ORT.


def _rotate_half_pattern(op, x, start1, end1, start2, end2):
    # Slice(input, starts, ends, axes, steps)
    x1 = op.Slice(x, start1, end1, [3], [1])
    x2 = op.Slice(x, start2, end2, [3], [1])
    minus_x2 = op.Neg(x2)
    rotated_x = op.Concat(minus_x2, x1, axis=-1)
    return rotated_x


class RotaryEmbeddingFusion(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__(name="RotaryEmbedding", as_function=True)

    def pattern(self, op, x, cos, sin, start1, end1, start2, end2):
        return x * cos + _rotate_half_pattern(op, x, start1, end1, start2, end2) * sin

    def check(self, op, x, start1, end1, start2, end2, **_):
        # x needs to be a 4D tensor with known last dimension size (== head_size) and known second dimension (num_heads)
        if x is None or x.shape is None or len(x.shape) != 4:
            return False
        if not isinstance(x.shape[1], int):
            return False
        head_size = x.shape[3]
        if not isinstance(head_size, int):
            return False
        half_head_size = head_size // 2

        # Check that x is being split into two equal halves of size half_head_size
        return (
            _ir_utils.is_singleton_value(start1, 0)
            and _ir_utils.is_singleton_value(end1, half_head_size)
            and _ir_utils.is_singleton_value(start2, half_head_size)
            and _ir_utils.is_singleton_value(end2, lambda x: x >= head_size)
        )

    def rewrite(self, op, x, cos, sin, **_):
        num_heads = x.shape[1]
        return op.RotaryEmbedding(
            x, cos, sin, interleaved=0, num_heads=num_heads, _domain="ai.onnxruntime.fusion"
        )


class PartialRotaryEmbeddingFusion(pattern.RewriteRuleClassBase):
    def pattern(self, op, x, end1, start2):
        x_part_1 = op.Slice(x, [0], end1, [3], [1])
        x_part_2 = op.Slice(x, start2, [9223372036854775807], [3], [1])
        x_part_1_rope = op.RotaryEmbedding(
            x_part_1,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _domain="com.microsoft",
            _outputs=["x_part_1_rope"],
        )
        return op.Concat(x_part_1_rope, x_part_2, axis=-1)

    def check(self, op, x, end1, start2, x_part_1_rope, **_):
        end1_value = _ir_utils.get_singleton_value(end1)
        start2_value = _ir_utils.get_singleton_value(start2)
        if not isinstance(end1_value, int) or not isinstance(start2_value, int):
            return False
        if end1_value != start2_value:
            return False
        rotary_embedding_attributes = x_part_1_rope.producer().attributes
        if "rotary_embedding_dim" in rotary_embedding_attributes:
            return False
        if (
            "interleaved" in rotary_embedding_attributes
            and rotary_embedding_attributes["interleaved"].value != 0
        ):
            return False
        return True

    def rewrite(self, op, x, end1, x_part_1_rope, **_):
        # Create a modified version of the RotaryEmbedding op:
        rotary_embedding_dim = _ir_utils.get_singleton_value(end1)
        original_node = x_part_1_rope.producer()
        inputs = list(original_node.inputs)
        inputs[0] = x
        attrs = dict(original_node.attributes)
        attrs["rotary_embedding_dim"] = rotary_embedding_dim
        return op.RotaryEmbedding(
            *inputs,
            **attrs,
            _domain="com.microsoft",
        )


_rule = RotaryEmbeddingFusion.rule()

_partial_embedding_rule = PartialRotaryEmbeddingFusion.rule()

rotary_embedding_rules = pattern.RewriteRuleSet([_rule])

partial_embedding_rules = pattern.RewriteRuleSet([_partial_embedding_rule])


def fuse_rotary_embedding(model: ir.Model) -> int:
    count = rotary_embedding_rules.apply_to_model(model)
    return count


def fuse_partial_rotary_embedding(model: ir.Model, debug: bool = False) -> int:
    count = partial_embedding_rules.apply_to_model(model)
    if count == 0 and debug:
        tracer = pattern.MatchingTracer()
        partial_embedding_rules.apply_to_model(model, tracer=tracer)
        tracer.report()
    return count
