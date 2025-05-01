# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

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

    def check(self, op, x, start1, end1, start2, end2, **_) -> pattern.MatchResult:  # type: ignore[name-defined]
        check_result = pattern.MatchResult()
        # x needs to be a 4D tensor with known last dimension size (== head_size) and known second dimension (num_heads)
        if x is None or x.shape is None or len(x.shape) != 4:
            return check_result.fail("Input is not a 4D tensor.", x)
        if not isinstance(x.shape[1], int):
            return check_result.fail("Input dimension 1 is not an integer.", x)
        head_size = x.shape[3]
        if not isinstance(head_size, int):
            return check_result.fail("Head size is not an integer.", x)
        half_head_size = head_size // 2

        # Check that x is being split into two equal halves of size half_head_size
        if not (
            _ir_utils.is_singleton_value(start1, 0)
            and _ir_utils.is_singleton_value(end1, half_head_size)
            and _ir_utils.is_singleton_value(start2, half_head_size)
            and _ir_utils.is_singleton_value(end2, lambda x: x >= head_size)
        ):
            return check_result.fail(
                "x is not being split into two equal halves of size half_head_size."
            )
        return check_result

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

    def check(self, op, x, end1, start2, x_part_1_rope, **_) -> pattern.MatchResult:  # type: ignore[name-defined]
        check_result = pattern.MatchResult()
        end1_value = _ir_utils.get_singleton_value(end1)
        start2_value = _ir_utils.get_singleton_value(start2)
        if not isinstance(end1_value, int) or not isinstance(start2_value, int):
            return check_result.fail(
                "The end1 value of first slice and start2 value of second slice are not integers."
            )
        if end1_value != start2_value:
            return check_result.fail(
                "The end1 value of first slice and start2 value of second slice are not equal."
            )
        rotary_embedding_attributes = x_part_1_rope.producer().attributes
        if "rotary_embedding_dim" in rotary_embedding_attributes:
            return check_result.fail("rotary_embedding_dim attribute already specified.")
        if (
            "interleaved" in rotary_embedding_attributes
            and rotary_embedding_attributes["interleaved"].value != 0
        ):
            return check_result.fail("interleaved is not equal to 0.")
        return check_result

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


fuse_rotary_embedding = _fusion_utils.apply_fusion_rules(rotary_embedding_rules)


fuse_partial_rotary_embedding = _fusion_utils.apply_fusion_rules(partial_embedding_rules)
