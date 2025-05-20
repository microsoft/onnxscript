# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np

import onnxscript.ir as ir
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
# inv_freq_values = inv_freq_expanded.reshape(1, -1)
# pos_id_range = np.arange(max_pos_id, dtype=np.float32).reshape(-1, 1)
# angles = np.matmul(pos_id_range, inv_freq_values)
# cos_value = np.cos(angles)
# sin_value = np.sin(angles)
# cos_2d = op.Constant(value=ir.tensor(cos_value))
# sin_2d = op.Constant(value=ir.tensor(sin_value))
#
# This produces cos/sin values in a form that can be used by ORT's custom ops.


class CosSinCacheFusion(pattern.RewriteRuleClassBase):
    def __init__(
        self,
        name: str,
        *,
        cast: bool = False,
        reshape: bool = False,
        const_freqs: bool = False,
    ):
        # This pattern makes use of shared Cos/Sin values. So, we can't remove the
        # matched nodes as part of the rewrite-step. We apply a separate final
        # pass to remove unused nodes.
        super().__init__(name, remove_nodes=False)
        # TODO: Determine what should be the default max_pos_id value
        self._max_pos_id = None
        # map from inv_freq to (cos, sin) values for transformed graph
        self._inv_freq_cos_sin_cache: dict[ir.Value, tuple[ir.Value, ir.Value]] = {}
        self._reshape = reshape
        self._cast = cast
        self._const_freqs = const_freqs

    @property
    def max_pos_id(self) -> int | None:
        return self._max_pos_id

    @max_pos_id.setter
    def max_pos_id(self, max_pos_id: int):
        self._max_pos_id = max_pos_id  # type: ignore[assignment]

    def _compute_const_freqs(self, op, angles: np.ndarray):
        """Compute cos/sin values when frequencies are constant."""
        cos_value = np.cos(angles)
        sin_value = np.sin(angles)
        cos_2d = op.Constant(value=ir.tensor(cos_value))
        sin_2d = op.Constant(value=ir.tensor(sin_value))
        return cos_2d, sin_2d

    def _compute_dynamic_freqs(self, op, inv_freq, position_ids, dtype):
        """Compute cos/sin values dynamically based on inv_freq and position_ids."""
        if self._max_pos_id is not None:
            # Use max_pos_id from the model metadata
            max_pos_id = self._max_pos_id
        elif position_ids.const_value is not None:
            # Calculate max_pos_id from the position_ids tensor
            max_pos_id = int(np.max(position_ids.const_value.numpy()))
        else:
            # Dynamically compute max_pos_id from position_ids using ONNX ops
            inv_freq = op.Reshape(inv_freq, op.Constant(value_ints=[1, -1]))
            max_pos_id = op.ReduceMax(position_ids, keepdims=0)
            max_pos_id = op.Add(max_pos_id, op.Constant(value_int=1))
            pos_id_range = op.Range(
                op.Constant(value_int=0),
                max_pos_id,
                op.Constant(value_int=1),
            )
            pos_id_range = op.Reshape(pos_id_range, op.Constant(value_ints=[-1, 1]))
            pos_id_range = op.Cast(pos_id_range, to=ir.DataType.FLOAT)
            # Compute angles and cos/sin values
            angles = op.MatMul(pos_id_range, inv_freq)
            cos_2d = op.Cos(angles)
            sin_2d = op.Sin(angles)
            return cos_2d, sin_2d

        # If we do not compute max_pos_id using ONNX ops, use inv_freq and position_ids
        # to compute angles and cos/sin values
        # Note: The one is added to max_pos_id as position_ids are 0-indexed
        # and the range of position ids should be [0, max_pos_id], max_pos_id inclusive.
        inv_freq_values = inv_freq.const_value.numpy().reshape(1, -1)
        pos_id_range = np.arange(max_pos_id + 1, dtype=np.float32).reshape(-1, 1)
        angles = np.matmul(pos_id_range, inv_freq_values)
        return self._compute_const_freqs(op, angles)

    def cleanup(self):
        self._inv_freq_cos_sin_cache.clear()

    def pattern(
        self, op, x, inv_freq, position_ids, interleaved, num_heads, freqs, dtype, extra_dims
    ):
        if not self._const_freqs:
            # Compute freqs from inv_freq and position_ids. In the _const_freqs case,
            # this computation has been constant-folded away and freqs is a constant.
            # B: batch size, S: sequence length, E: embedding dimension
            # position_ids: [B, S] or [S]
            # inv_freq: [1, E, 1]
            position_ids_expanded = op.Unsqueeze(
                position_ids, extra_dims
            )  # [B, S] | [S] => [B, 1, S]
            position_ids_expanded = op.Cast(position_ids_expanded, to=ir.DataType.FLOAT)
            # if self._reshape:
            #     position_ids_expanded = op.Expand(position_ids_expanded, _allow_other_inputs=True)
            #     position_ids_expanded = op.Reshape(position_ids_expanded, _allow_other_inputs=True)
            freqs = op.MatMul(inv_freq, position_ids_expanded)  # [B, E, S]
        # if self._reshape:
        #     freqs = op.Reshape(freqs, freqs_3d_shape)  # redundant reshape
        freqs = op.Transpose(freqs, perm=[0, 2, 1])  # [B, S, E]
        emb = op.Concat(freqs, freqs, axis=-1)
        cos = op.Cos(emb)
        if self._cast:
            cos = op.Cast(cos, to=dtype)
        sin = op.Sin(emb)
        if self._cast:
            sin = op.Cast(sin, to=dtype)
        cos_4d = op.Unsqueeze(cos, 1)  # convert
        sin_4d = op.Unsqueeze(sin, 1)
        return op.RotaryEmbedding(
            x,
            cos_4d,
            sin_4d,
            interleaved=interleaved,
            num_heads=num_heads,
            _domain="ai.onnxruntime.fusion",
        )

    def check(
        self, context, inv_freq, position_ids, freqs, extra_dims, **_
    ) -> pattern.MatchResult:  # type: ignore[name-defined]
        check_result = pattern.MatchResult()
        # TODO(rama): handle redundant reshape/expand
        if self._const_freqs:
            if (freqs.const_value is None) or not _ir_utils.has_rank(freqs, 3):
                return check_result.fail("freqs is not a constant or not 3D.", freqs)
            else:
                return check_result
        if (
            _ir_utils.has_rank(position_ids, 2) and _ir_utils.is_singleton_value(extra_dims, 1)
        ) or (
            _ir_utils.has_rank(position_ids, 1) and _ir_utils.is_1d_value(extra_dims, [0, 1])
        ):
            pass
        else:
            return check_result.fail("position_ids is not a 1D or 2D tensor.", position_ids)
        if not _ir_utils.has_rank(inv_freq, 3):
            return check_result.fail("inv_freq is not 3D.", inv_freq)
        inv_freq_shape = inv_freq.shape
        if inv_freq.const_value is None:  # TODO: should this be inv_freq_shape?
            return check_result.fail("inv_freq is not a constant.", inv_freq)
        if inv_freq_shape[0] != 1 or inv_freq_shape[2] != 1:
            return check_result.fail("inv_freq is not of shape [1, ., 1].", inv_freq)
        return check_result

    def rewrite(
        self, op, x, inv_freq, position_ids, interleaved, num_heads, freqs, dtype, **_
    ):
        if inv_freq in self._inv_freq_cos_sin_cache:
            cos_2d, sin_2d = self._inv_freq_cos_sin_cache[inv_freq]
        else:
            # Compute cos/sin values based on whether frequencies are constant
            if self._const_freqs:
                cos_2d, sin_2d = self._compute_const_freqs(op, freqs.const_value.numpy())
            else:
                cos_2d, sin_2d = self._compute_dynamic_freqs(op, inv_freq, position_ids, dtype)
            if self._cast:
                cos_2d = op.Cast(cos_2d, to=dtype)
                sin_2d = op.Cast(sin_2d, to=dtype)
            self._inv_freq_cos_sin_cache[inv_freq] = (cos_2d, sin_2d)
        if _ir_utils.has_rank(position_ids, 1):
            zero_1d = op.Constant(value_ints=[0])
            position_ids = op.Unsqueeze(position_ids, zero_1d)
        return op.RotaryEmbedding(
            x,
            position_ids,
            cos_2d,
            sin_2d,
            interleaved=interleaved,
            num_heads=num_heads,
            _domain="com.microsoft",
        )


_cast_const_freqs = CosSinCacheFusion.rule(
    "CosSinCache_cast_const_freqs", cast=True, const_freqs=True
)
_cast = CosSinCacheFusion.rule("CosSinCache_cast", cast=True, const_freqs=False)
_const_freqs = CosSinCacheFusion.rule("CosSinCache_const_freqs", cast=False, const_freqs=True)
_basic = CosSinCacheFusion.rule("CosSinCache", cast=False)

cos_sin_cache_rules = pattern.RewriteRuleSet([_cast, _cast_const_freqs, _const_freqs, _basic])


fuse_cos_sin_cache = _fusion_utils.apply_fusion_rules(cos_sin_cache_rules)
