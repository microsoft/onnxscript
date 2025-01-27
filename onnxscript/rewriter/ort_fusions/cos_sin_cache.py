# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import numpy as np

import onnxscript.ir as ir
from onnxscript.optimizer import remove_unused_nodes
from onnxscript.rewriter import _ir_utils, pattern

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

# TODO: To apply the pattern-rewrite, we need to know the maximum position id.
# Need to find a way to get this information from the model or its config.


class CosSinCacheFusion(pattern.RewriteRuleClassBase):
    def __init__(
        self,
        name: str,
        max_pos_id: int,
        *,
        cast: bool = False,
        reshape: bool = False,
        const_freqs: bool = False,
    ):
        # This pattern makes use of shared Cos/Sin values. So, we can't remove the
        # matched nodes as part of the rewrite-step. We apply a separate final
        # pass to remove unused nodes.
        super().__init__(name, remove_nodes=False)
        self._max_pos_id = max_pos_id
        # map from inv_freq to (cos, sin) values for transformed graph
        self._inv_freq_cos_sin_cache: dict[ir.Value, tuple[ir.Value, ir.Value]] = {}
        self._reshape = reshape
        self._cast = cast
        self._const_freqs = const_freqs

    def cleanup(self):
        self._inv_freq_cos_sin_cache.clear()

    def pattern(self, op, x, inv_freq, position_ids, interleaved, num_heads, freqs, dtype):
        if not self._const_freqs:
            # Compute freqs from inv_freq and position_ids. In the _const_freqs case,
            # this computation has been constant-folded away and freqs is a constant.
            # B: batch size, S: sequence length, E: embedding dimension
            # position_ids: [B, S]
            # inv_freq: [1, E, 1]
            position_ids_expanded = op.Unsqueeze(position_ids, 1)  # [B, S] => [B, 1, S]
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

    def check(self, context, inv_freq, position_ids, freqs, **_):
        # TODO(rama): handle redundant reshape/expand
        if self._const_freqs:
            return (freqs.const_value is not None) and _ir_utils.has_rank(freqs, 3)
        if not _ir_utils.has_rank(position_ids, 2):
            return False
        if not _ir_utils.has_rank(inv_freq, 3):
            return False
        inv_freq_shape = inv_freq.shape
        if inv_freq.const_value is None:  # TODO: should this be inv_freq_shape?
            return False
        return inv_freq_shape[0] == 1 and inv_freq_shape[2] == 1

    def rewrite(
        self, op, x, inv_freq, position_ids, interleaved, num_heads, freqs, dtype, **_
    ):
        if inv_freq in self._inv_freq_cos_sin_cache:
            cos_2d, sin_2d = self._inv_freq_cos_sin_cache[inv_freq]
        else:
            if self._const_freqs:
                angles = freqs.const_value.numpy()
            else:
                inv_freq_values = inv_freq.const_value.numpy().reshape(1, -1)
                pos_id_range = np.arange(self._max_pos_id, dtype=np.float32).reshape(-1, 1)
                angles = np.matmul(pos_id_range, inv_freq_values)
            cos_value = np.cos(angles)
            sin_value = np.sin(angles)
            cos_2d = op.Constant(value=ir.tensor(cos_value))
            sin_2d = op.Constant(value=ir.tensor(sin_value))
            if self._cast:
                cos_2d = op.Cast(cos_2d, to=dtype)
                sin_2d = op.Cast(sin_2d, to=dtype)
            self._inv_freq_cos_sin_cache[inv_freq] = (cos_2d, sin_2d)
        return op.RotaryEmbedding(
            x,
            position_ids,
            cos_2d,
            sin_2d,
            interleaved=interleaved,
            num_heads=num_heads,
            _domain="com.microsoft",
        )


_cast = CosSinCacheFusion.rule("CosSinCache", 2048, cast=True, const_freqs=True)
_no_cast = CosSinCacheFusion.rule("CosSinCache", 2048, cast=False)

cos_sin_cache_rules = pattern.RewriteRuleSet([_cast, _no_cast])

debug: bool = True


def fuse_cos_sin_cache(model: ir.Model) -> int:
    count = cos_sin_cache_rules.apply_to_model(model)
    if count == 0 and debug:
        cos_sin_cache_rules.apply_to_model(model, debug=True)
    else:
        print(f"CosSinCache count: {count}")
        remove_unused_nodes(model)
    return count
