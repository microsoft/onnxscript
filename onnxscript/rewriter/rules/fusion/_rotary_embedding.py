# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

# Fusions for RotaryEmbedding:
# Fuse computation patterns seen in HF transformer models for RotaryEmbedding
# and map them to ONNX opset 23 RotaryEmbedding op.

# Basic pattern: For example, see
# https://github.com/huggingface/transformers/blob/541bed22d6e4f97946a3a7d74f7e1a353e58643b/src/transformers/models/llama/modeling_llama.py#L104
#    def rotate_half(x):
#        """Rotates half the hidden dims of the input."""
#        x1 = x[..., : x.shape[-1] // 2]
#        x2 = x[..., x.shape[-1] // 2 :]
#        return torch.cat((-x2, x1), dim=-1)
# and
#        q_embed = (q * cos) + (rotate_half(q) * sin)


def _rotate_half_pattern(op, x, start1, end1, start2, end2):
    # Slice(input, starts, ends, axes, steps)
    x1 = op.Slice(x, start1, end1, [3], [1])
    x2 = op.Slice(x, start2, end2, [3], [1])
    minus_x2 = op.Neg(x2)
    rotated_x = op.Concat(minus_x2, x1, axis=-1)
    return rotated_x


class RotaryEmbedding23Fusion(pattern.RewriteRuleClassBase):
    def __init__(self):
        super().__init__(name="RotaryEmbedding23")

    def pattern(self, op, x, cos, sin, start1, end1, start2, end2):
        return x * cos + _rotate_half_pattern(op, x, start1, end1, start2, end2) * sin

    def check(self, op, x, start1, end1, start2, end2, **_) -> pattern.MatchResult:  # type: ignore[name-defined]
        check_result = pattern.MatchResult()
        # x needs to be a 4D tensor with known last dimension size (== head_size) and known second dimension (num_heads)
        if x is None or x.shape is None or len(x.shape) != 4:
            return check_result.fail("Input is not known to be a 4D tensor.", x)
        if not isinstance(x.shape[1], int):
            return check_result.fail("Input dimension 1 (num_heads) is not static.", x)
        head_size = x.shape[3]
        if not isinstance(head_size, int):
            return check_result.fail("Head size is not static.", x)
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
            x,
            cos,
            sin,
            interleaved=0,
            num_heads=num_heads,
        )


# Extensions for partial rotary embedding fusion: with partial rotary embedding,
# embedding is applied only to the first part of the input, and the second part is left unchanged,
# as captured in the pattern below.

MAX_INT64 = 9223372036854775807


class PartialRotaryEmbedding23Fusion(pattern.RewriteRuleClassBase):
    def pattern(self, op, x, end1, start2):
        x_part_1 = op.Slice(x, [0], end1, [3], [1])
        x_part_2 = op.Slice(x, start2, [MAX_INT64], [3], [1])
        x_part_1_rope = op.RotaryEmbedding(
            x_part_1,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["x_part_1_rope"],
        )
        return op.Concat(x_part_1_rope, x_part_2, axis=-1)

    def check(self, op, x, end1, start2, x_part_1_rope, **_) -> pattern.MatchResult:  # type: ignore[name-defined]
        check_result = pattern.MatchResult()
        end1_value = _ir_utils.get_singleton_value(end1)
        start2_value = _ir_utils.get_singleton_value(start2)
        if not isinstance(end1_value, int) or not isinstance(start2_value, int):
            return check_result.fail("Unable to validate slice start/end values.")
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
        )


_rule = RotaryEmbedding23Fusion.rule()

_partial_embedding_rule = PartialRotaryEmbedding23Fusion.rule()

rotary_embedding_rules = pattern.RewriteRuleSet([_rule])

partial_embedding_rules = pattern.RewriteRuleSet([_partial_embedding_rule])

fuse_rotary_embedding = _fusion_utils.apply_fusion_rules(rotary_embedding_rules)

fuse_partial_rotary_embedding = _fusion_utils.apply_fusion_rules(partial_embedding_rules)
