# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnx_ir as ir

from onnxscript.rewriter import _fusion_utils, _ir_utils, pattern

# Rewrite the computation of cos/sin cache into the form expected by ORT's custom ops.

# We match against the following code pattern:
# Original code (from transformers) for computing cos/sin cache for RoPE:
# https://github.com/huggingface/transformers/blob/0ade1caa356dce6b70ef8293addeb0898f177206/src/transformers/models/llama/modeling_llama.py#L135
# position_ids_expanded = position_ids[:, None, :].float()
# freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
# emb = torch.cat((freqs, freqs), dim=-1)
# cos = emb.cos()
# sin = emb.sin()
#
# We rewrite this pattern into the following form:
# angles = position_ids.reshape(-1, 1).float() @ inv_freq.reshape(1, -1)
# cos_2d = cos(angles)
# sin_2d = sin(angles)
# cache_indices = arange(position_ids.numel()).reshape(position_ids.shape)
#
# This produces cos/sin values in a form that can be used by ORT's custom ops.


class CosSinCacheFusion(pattern.RewriteRuleClassBase):
    def __init__(
        self,
        name: str,
        *,
        cast: bool = False,
        const_freqs: bool = False,
        concat_after_trig: bool = False,
    ):
        # This pattern makes use of shared Cos/Sin values. So, we can't remove the
        # matched nodes as part of the rewrite-step. We apply a separate final
        # pass to remove unused nodes.
        super().__init__(name, remove_nodes=False)
        self._cast = cast
        self._const_freqs = const_freqs
        self._concat_after_trig = concat_after_trig

    def pattern(
        self,
        op,
        x,
        inv_freq,
        position_ids,
        interleaved,
        num_heads,
        freqs,
        dtype,
        extra_dims,
    ):
        if not self._const_freqs:
            # Compute freqs from inv_freq and position_ids. In the _const_freqs case,
            # this computation has been constant-folded away and freqs is a constant.
            # B: batch size, S: sequence length, E: embedding dimension
            # position_ids: [B, S] or [S]
            # inv_freq: [1, E, 1]
            position_ids_expanded = pattern.OrValue(
                [
                    op.Cast(op.Unsqueeze(position_ids, extra_dims), to=ir.DataType.FLOAT),
                    op.Unsqueeze(op.Cast(position_ids, to=ir.DataType.FLOAT), extra_dims),
                ]
            )
            # inv_freq may optionally be expanded to shape [B, E, 1]
            inv_freq = pattern.OrValue(
                [
                    op.Expand(inv_freq, pattern.ANY_VALUE, _outputs=["expanded_inv_freq"]),
                    inv_freq,
                ]
            )
            freqs = op.MatMul(inv_freq, position_ids_expanded)  # [B, E, S]
        freqs = op.Transpose(freqs, perm=[0, 2, 1])  # [B, S, E]
        emb = freqs if self._concat_after_trig else op.Concat(freqs, freqs, axis=-1)
        cos = op.Cos(emb)
        if self._cast:
            cos = op.Cast(cos, to=dtype, _outputs=["cos_cast"])
        sin = op.Sin(emb)
        if self._cast:
            sin = op.Cast(sin, to=dtype, _outputs=["sin_cast"])
        if self._concat_after_trig:
            cos = op.Concat(cos, cos, axis=-1)
            sin = op.Concat(sin, sin, axis=-1)
        cos_4d = op.Unsqueeze(cos, [1])  # convert
        sin_4d = op.Unsqueeze(sin, [1])
        return op.RotaryEmbedding(
            x,
            cos_4d,
            sin_4d,
            interleaved=interleaved,
            num_heads=num_heads,
            _domain="ai.onnxruntime._fusion",
        )

    def check(
        self,
        context,
        x,
        inv_freq,
        position_ids,
        freqs,
        extra_dims,
        interleaved,
        num_heads,
        dtype,
        expanded_inv_freq=None,
        cos_cast=None,
        sin_cast=None,
        **_,
    ) -> pattern.MatchResult:  # type: ignore[name-defined]
        check_result = pattern.MatchResult()
        imports = context.graph_or_function.opset_imports
        if imports.get("", context.model.opset_imports.get("", 0)) < 15:
            return check_result.fail("Cache conversion requires bounded Shape (opset 15+).")
        if (
            not _ir_utils.has_rank(x, 4)
            or x.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16)
            or not isinstance(x.shape[3], int)
            or x.shape[3] <= 0
            or x.shape[3] % 2
        ):
            return check_result.fail(
                "ORT rotary cache requires FLOAT/FLOAT16 rank-four input and an even head size."
            )
        if interleaved.is_ref() or num_heads.is_ref() or interleaved.as_int() != 0:
            return check_result.fail(
                "Rotary attributes must specify concrete noninterleaved layout."
            )
        if (
            not isinstance(x.shape[1], int)
            or x.shape[1] <= 0
            or num_heads.as_int() != x.shape[1]
        ):
            return check_result.fail("Rotary head count must match the input layout.")
        if self._cast:
            if dtype.is_ref() or dtype.as_int() != x.dtype:
                return check_result.fail("Cache Cast type must match the rotary data type.")
            attrs = cos_cast.producer().attributes
            other_attrs = sin_cast.producer().attributes
            if any(attr.is_ref() for attr in (*attrs.values(), *other_attrs.values())):
                return check_result.fail("Cache Cast attributes must be concrete.")
            if set(attrs) != set(other_attrs) or any(
                attr.value != other_attrs[name].value for name, attr in attrs.items()
            ):
                return check_result.fail("Cosine and sine Cast semantics must match.")
            self._cast_attributes = dict(attrs)
        elif x.dtype != ir.DataType.FLOAT:
            return check_result.fail("Uncast caches require FLOAT rotary input.")
        self._shape_inputs = tuple(context.root.graph.inputs)
        if self._const_freqs:
            if _ir_utils.get_const_value(freqs) is None or not _ir_utils.has_rank(freqs, 3):
                return check_result.fail("freqs is not a constant or not 3D.", freqs)
            if (
                freqs.dtype != ir.DataType.FLOAT
                or freqs.shape[1] != x.shape[3] // 2
                or not (freqs.shape[0] == 1 or _ir_utils.same_dim(freqs.shape[0], x.shape[0]))
                or not (freqs.shape[2] == 1 or _ir_utils.same_dim(freqs.shape[2], x.shape[2]))
            ):
                return check_result.fail(
                    "Constant frequencies must broadcast over the rotary batch and sequence."
                )
            return check_result
        if (
            _ir_utils.has_rank(position_ids, 2) and _ir_utils.is_singleton_value(extra_dims, 1)
        ) or (
            _ir_utils.has_rank(position_ids, 1) and _ir_utils.is_1d_value(extra_dims, [0, 1])
        ):
            pass
        else:
            return check_result.fail("position_ids is not a 1D or 2D tensor.", position_ids)
        if not (
            (
                position_ids.shape[-1] == 1
                or _ir_utils.same_dim(position_ids.shape[-1], x.shape[2])
            )
            and (
                len(position_ids.shape) == 1
                or position_ids.shape[0] == 1
                or _ir_utils.same_dim(position_ids.shape[0], x.shape[0])
            )
        ):
            return check_result.fail(
                "Position dimensions must broadcast into the rotary data shape."
            )
        if not _ir_utils.has_rank(inv_freq, 3):
            return check_result.fail("inv_freq is not 3D.", inv_freq)
        inv_freq_shape = inv_freq.shape
        if expanded_inv_freq is not None:
            if not _ir_utils.has_rank(expanded_inv_freq, 3):
                return check_result.fail("expanded_inv_freq is not 3D.", expanded_inv_freq)
            if not (
                _ir_utils.same_dim(expanded_inv_freq.shape[1], inv_freq_shape[1])
                and expanded_inv_freq.shape[2] == 1
                and (
                    expanded_inv_freq.shape[0] == 1
                    or _ir_utils.same_dim(expanded_inv_freq.shape[0], x.shape[0])
                )
            ):
                return check_result.fail(
                    "Inverse-frequency expansion must only broadcast the batch."
                )
        if _ir_utils.get_const_value(inv_freq) is None:
            return check_result.fail("inv_freq is not a constant.", inv_freq)
        if (
            inv_freq.dtype != ir.DataType.FLOAT
            or inv_freq_shape[0] != 1
            or inv_freq_shape[2] != 1
            or inv_freq_shape[1] != x.shape[3] // 2
        ):
            return check_result.fail("inv_freq is not of shape [1, ., 1].", inv_freq)
        return check_result

    def rewrite(
        self, op, x, inv_freq, position_ids, interleaved, num_heads, freqs, dtype, **_
    ):
        shape_source = x
        indices: list[int] | None = [0, 2]
        candidate = x
        # Avoid keeping attention/partial-rotary intermediates alive just for Shape.
        while (
            (producer := candidate.producer()) is not None
            and producer.domain == ""
            and producer.op_type in ("Transpose", "Reshape", "Split", "Slice")
        ):
            candidate = producer.inputs[0]
            if candidate is None or candidate.shape is None:
                break
            if len(candidate.shape) in (3, 4):
                sequence_axis = next(
                    (
                        axis
                        for axis in (1, 2)
                        if _ir_utils.same_dim(candidate.shape[axis], x.shape[2])
                    ),
                    None,
                )
                if sequence_axis is not None and _ir_utils.same_dim(
                    candidate.shape[0], x.shape[0]
                ):
                    shape_source, indices = candidate, [0, sequence_axis]
            elif (
                len(candidate.shape) == 2
                and x.shape[0] == 1
                and _ir_utils.same_dim(candidate.shape[0], x.shape[2])
            ):
                shape_source, indices = candidate, None
        for candidate in self._shape_inputs:
            if (
                candidate.shape is not None
                and len(candidate.shape) >= 2
                and _ir_utils.same_dim(candidate.shape[0], x.shape[0])
                and _ir_utils.same_dim(candidate.shape[1], x.shape[2])
            ):
                shape_source, indices = candidate, [0, 1]
                break
        if indices is None:
            batch_sequence = op.Concat(
                op.Constant(value_ints=[1]), op.Shape(shape_source, start=0, end=1), axis=0
            )
        else:
            batch_sequence = op.Gather(
                op.Shape(shape_source), op.Constant(value_ints=indices), axis=0
            )
        if self._const_freqs:
            angles = op.Transpose(freqs, perm=[0, 2, 1])
            shape = op.Concat(
                batch_sequence, op.Constant(value_ints=[x.shape[3] // 2]), axis=0
            )
            angles = op.Expand(angles, shape)
            angles = op.Reshape(angles, op.Constant(value_ints=[-1, x.shape[3] // 2]))
        else:
            position_ids = op.Expand(position_ids, batch_sequence)
            positions = op.Reshape(position_ids, op.Constant(value_ints=[-1, 1]))
            positions = op.Cast(positions, to=ir.DataType.FLOAT)
            inverse = op.Reshape(inv_freq, op.Constant(value_ints=[1, -1]))
            angles = op.MatMul(positions, inverse)
        cos_2d, sin_2d = op.Cos(angles), op.Sin(angles)
        if self._cast:
            cos_2d = op.Cast(cos_2d, **self._cast_attributes)
            sin_2d = op.Cast(sin_2d, **self._cast_attributes)
        cache_count = op.ReduceProd(batch_sequence, keepdims=0)
        if not isinstance(x.shape[0], int) or x.shape[0] == 0:
            # ORT requires at least sequence_length cache rows, even for an empty batch.
            sequence = op.Gather(batch_sequence, op.Constant(value_int=1), axis=0)
            extra_rows = op.Max(op.Sub(sequence, cache_count), op.Constant(value_int=0))
            pads = op.Concat(
                op.Constant(value_ints=[0, 0]),
                op.Unsqueeze(extra_rows, op.Constant(value_ints=[0])),
                op.Constant(value_ints=[0]),
                axis=0,
            )
            cos_2d = op.Pad(cos_2d, pads)
            sin_2d = op.Pad(sin_2d, pads)
        cache_indices = op.Range(
            op.Constant(value_int=0), cache_count, op.Constant(value_int=1)
        )
        cache_indices = op.Reshape(cache_indices, batch_sequence, allowzero=1)
        return op.RotaryEmbedding(
            x,
            cache_indices,
            cos_2d,
            sin_2d,
            interleaved=interleaved,
            num_heads=num_heads,
            _domain="com.microsoft",
        )


_cast_const_freqs = CosSinCacheFusion.rule(
    "FuseCosSinCacheFromConstantFrequenciesWithCast", cast=True, const_freqs=True
)
_cast = CosSinCacheFusion.rule("FuseCosSinCacheWithCast", cast=True, const_freqs=False)
_const_freqs = CosSinCacheFusion.rule(
    "FuseCosSinCacheFromConstantFrequencies", cast=False, const_freqs=True
)
_basic = CosSinCacheFusion.rule("FuseCosSinCache", cast=False)

cos_sin_cache_rules = pattern.RewriteRuleSet(
    [
        _cast,
        _cast_const_freqs,
        _const_freqs,
        _basic,
        *(
            CosSinCacheFusion.rule(
                "FuseCosSinCacheWithRepeatedTrig"
                + ("AndCast" if cast else "")
                + ("FromConstantFrequencies" if const_freqs else ""),
                cast=cast,
                const_freqs=const_freqs,
                concat_after_trig=True,
            )
            for cast in (False, True)
            for const_freqs in (False, True)
        ),
    ]
)


fuse_cos_sin_cache = _fusion_utils.apply_fusion_rules(cos_sin_cache_rules)
