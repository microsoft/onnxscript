# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnx_ir as ir

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


def _rotary_split_size(x, split_sizes, split_axis):
    if (
        x.shape is None
        or len(x.shape) != 4
        or not isinstance(x.shape[3], int)
        or split_axis is None
        or split_axis.is_ref()
        or split_axis.as_int() not in (3, -1)
    ):
        return None
    sizes = _ir_utils.get_numpy_value(split_sizes)
    if (
        sizes is None
        or split_sizes.dtype != ir.DataType.INT64
        or sizes.ndim != 1
        or sizes.size != 2
        or any(size <= 0 for size in sizes)
        or sum(int(size) for size in sizes) != x.shape[3]
    ):
        return None
    return int(sizes[0])


def _rotate_half_pattern(
    op, x, start1, end1, start2, end2, split_sizes=None, split_axis=None, use_split=False
):
    # Slice(input, starts, ends, axes, steps)
    if use_split:
        x1, x2 = op.Split(x, split_sizes, axis=split_axis, _outputs=2)
    else:
        x1 = op.Slice(x, start1, end1, [3], [1])
        x2 = op.Slice(x, start2, end2, [3], [1])
    minus_x2 = op.Neg(x2)
    rotated_x = op.Concat(minus_x2, x1, axis=-1)
    return rotated_x


class RotaryEmbedding23Fusion(pattern.RewriteRuleClassBase):
    def __init__(self, use_split=False, concat_after_trig=False):
        self._use_split = use_split
        self._concat_after_trig = concat_after_trig
        name = f"FuseRotaryEmbedding23From{'Split' if use_split else 'Slices'}"
        if concat_after_trig:
            name += "WithRepeatedTrig"
        super().__init__(name=name, remove_nodes=False)

    def pattern(
        self,
        op,
        x,
        freqs,
        start1,
        end1,
        start2,
        end2,
        one1,
        one2,
        split_sizes=None,
        split_axis=None,
    ):
        if self._concat_after_trig:
            cos = op.Cos(freqs)
            sin = op.Sin(freqs)
            cos = op.Concat(cos, cos, axis=-1)
            sin = op.Concat(sin, sin, axis=-1)
        else:
            freqs_repeated = op.Concat(freqs, freqs, axis=-1)
            cos = op.Cos(freqs_repeated)
            sin = op.Sin(freqs_repeated)
        cos_4d = op.Unsqueeze(cos, one1)
        sin_4d = op.Unsqueeze(sin, one2)
        rotated = _rotate_half_pattern(
            op, x, start1, end1, start2, end2, split_sizes, split_axis, self._use_split
        )
        return x * cos_4d + rotated * sin_4d

    def check(
        self,
        op,
        x,
        freqs,
        one1,
        one2,
        start1=None,
        end1=None,
        start2=None,
        end2=None,
        split_sizes=None,
        split_axis=None,
        **_,
    ) -> pattern.MatchResult:  # type: ignore[name-defined]
        check_result = pattern.MatchResult()
        version = op.graph_or_function.opset_imports.get("", op.model.opset_imports.get("", 0))
        if version < 23:
            return check_result.fail(
                "Standard RotaryEmbedding requires ONNX opset 23 or newer."
            )

        if not _ir_utils.is_singleton_value(one1, 1):
            return check_result.fail("Unsqueeze axes is not [1]", one1)
        if not _ir_utils.is_singleton_value(one2, 1):
            return check_result.fail("Unsqueeze axes is not [1]", one2)

        # x needs to be a 4D tensor with known last dimension size (== head_size) and known second dimension (num_heads)
        if x is None or x.shape is None or len(x.shape) != 4:
            return check_result.fail("Input is not known to be a 4D tensor.", x)
        if not isinstance(x.shape[1], int) or x.shape[1] <= 0:
            return check_result.fail("Input dimension 1 (num_heads) is not static.", x)
        head_size = x.shape[3]
        if not isinstance(head_size, int):
            return check_result.fail("Head size is not static.", x)
        if head_size <= 0 or head_size % 2:
            return check_result.fail("Rotary head size must be a positive even integer.", x)
        if x.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16):
            return check_result.fail(
                "Standard RotaryEmbedding does not support this input type.", x
            )
        half_head_size = head_size // 2
        if not (
            _ir_utils.has_rank(freqs, 3)
            and _ir_utils.same_dim(freqs.shape[0], x.shape[0])
            and _ir_utils.same_dim(freqs.shape[1], x.shape[2])
            and freqs.shape[2] == half_head_size
        ):
            return check_result.fail(
                "Frequency dimensions must match the rotary batch, sequence, and half head."
            )

        # Check that x is being split into two equal halves of size half_head_size
        if self._use_split:
            split_size = _rotary_split_size(x, split_sizes, split_axis)
            if split_size != half_head_size or head_size != 2 * split_size:
                return check_result.fail("Split does not select equal rotary halves.")
        elif not (
            _ir_utils.is_singleton_value(start1, 0)
            and _ir_utils.is_singleton_value(end1, half_head_size)
            and _ir_utils.is_singleton_value(start2, half_head_size)
            and _ir_utils.is_singleton_value(end2, lambda x: x >= head_size)
        ):
            return check_result.fail(
                "x is not being split into two equal halves of size half_head_size."
            )
        return check_result

    def rewrite(self, op, x, freqs, **_):
        num_heads = x.shape[1]
        cos = op.Cos(freqs)
        sin = op.Sin(freqs)
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
    def __init__(self, use_split=False):
        super().__init__(
            f"FusePartialRotaryEmbedding23From{'Split' if use_split else 'Slices'}"
        )
        self._use_split = use_split

    def pattern(self, op, x, end1, start2, split_sizes=None, split_axis=None):
        if self._use_split:
            x_part_1, x_part_2 = op.Split(x, split_sizes, axis=split_axis, _outputs=2)
        else:
            x_part_1 = op.Slice(x, [0], end1, [3], [1])
            x_part_2 = op.Slice(x, start2, [MAX_INT64], [3], [1])
        x_part_1_rope = op.RotaryEmbedding(
            x_part_1,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["x_part_1_rope"],
        )
        return op.Concat(x_part_1_rope, x_part_2, axis=-1)

    def check(
        self,
        op,
        x,
        x_part_1_rope,
        end1=None,
        start2=None,
        split_sizes=None,
        split_axis=None,
        **_,
    ) -> pattern.MatchResult:  # type: ignore[name-defined]
        check_result = pattern.MatchResult()
        version = op.graph_or_function.opset_imports.get("", op.model.opset_imports.get("", 0))
        if version < 23:
            return check_result.fail(
                "Standard RotaryEmbedding requires ONNX opset 23 or newer."
            )
        if self._use_split:
            end1_value = start2_value = _rotary_split_size(x, split_sizes, split_axis)
        else:
            end1_value = _ir_utils.get_singleton_value(end1)
            start2_value = _ir_utils.get_singleton_value(start2)
        if not isinstance(end1_value, int) or not isinstance(start2_value, int):
            return check_result.fail("Unable to validate slice start/end values.")
        if end1_value != start2_value:
            return check_result.fail(
                "The end1 value of first slice and start2 value of second slice are not equal."
            )
        if not (
            _ir_utils.has_rank(x, 4)
            and isinstance(x.shape[3], int)
            and 0 < end1_value <= x.shape[3]
            and end1_value % 2 == 0
        ):
            return check_result.fail(
                "Partial rotary dimension must be positive, even, and within the input head."
            )
        rotary_embedding_attributes = x_part_1_rope.producer().attributes
        if any(attr.is_ref() for attr in rotary_embedding_attributes.values()):
            return check_result.fail("Rotary attributes must be concrete for partial fusion.")
        if "rotary_embedding_dim" in rotary_embedding_attributes:
            return check_result.fail("rotary_embedding_dim attribute already specified.")
        if (
            "interleaved" in rotary_embedding_attributes
            and rotary_embedding_attributes["interleaved"].value != 0
        ):
            return check_result.fail("interleaved is not equal to 0.")
        self._rotary_dimension = end1_value
        return check_result

    def rewrite(self, op, x, x_part_1_rope, **_):
        # Create a modified version of the RotaryEmbedding op:
        rotary_embedding_dim = self._rotary_dimension
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

rotary_embedding_rules = pattern.RewriteRuleSet(
    [
        _rule,
        RotaryEmbedding23Fusion.rule(True),
        RotaryEmbedding23Fusion.rule(False, True),
        RotaryEmbedding23Fusion.rule(True, True),
    ]
)

partial_embedding_rules = pattern.RewriteRuleSet(
    [_partial_embedding_rule, PartialRotaryEmbedding23Fusion.rule(True)]
)

fuse_rotary_embedding = _fusion_utils.apply_fusion_rules(rotary_embedding_rules)

fuse_partial_rotary_embedding = _fusion_utils.apply_fusion_rules(partial_embedding_rules)
