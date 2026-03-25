# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rules for fusing RotaryEmbedding + Attention into GroupQueryAttention.

The standard decoder layer pattern applies RotaryEmbedding to Q and K
before feeding them into the ONNX ``Attention`` op (with KV cache).
The ``com.microsoft::GroupQueryAttention`` custom op fuses rotary
embedding into the attention kernel and replaces the explicit attention
bias with ``seqlens_k`` / ``total_sequence_length`` inputs computed
from the ``attention_mask`` graph input.

A second rule (``PackQKVForGQA``) runs after the GQA fusion and
consolidates separate Q, K, V projection MatMuls into a single packed
MatMul when they share the same hidden_states input.  The packed QKV
tensor is passed in the ``query`` slot of ``GroupQueryAttention`` with
``key`` and ``value`` set to ``None``.  Models with QK norm (e.g.
Qwen3) are unaffected because the Q/K projections are followed by a
normalization op, so the pattern does not match.

These rules are **not applied by default**.  Apply them post-export::

    from mobius.rewrite_rules import group_query_attention_rules
    from onnxscript.rewriter import rewrite

    model = build("Qwen/Qwen3-0.6B")
    rewrite(model, pattern_rewrite_rules=group_query_attention_rules())
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter._basics import MatchFailureError, MatchResult
from onnxscript.rewriter._rewrite_rule import (
    RewriteRuleClassBase,
    RewriteRuleSet,
)


class RotaryAttentionToGQA(RewriteRuleClassBase):
    """Replace RotaryEmbedding + Attention with GroupQueryAttention.

    **Matched pattern:**

    .. code-block:: text

        q_rot = RotaryEmbedding(q_pre, cos, sin)
        k_rot = RotaryEmbedding(k_pre, cos, sin)
        attn_out, present_key, present_value = Attention(
            q_rot, k_rot, v, attention_bias, past_key, past_value,
        )

    Where ``past_key`` and ``past_value`` are graph inputs (decoder
    attention with KV cache), and ``cos`` / ``sin`` are position-gathered
    from rotary cache tables via ``Gather``.

    **Replacement:**

    .. code-block:: text

        seqlens_k = Cast(ReduceSum(attention_mask, axis=1) - 1, INT32)
        total_seq_len = Cast(Shape(attention_mask)[1], INT32)
        attn_out, present_key, present_value = GroupQueryAttention(
            q_pre, k_pre, v, past_key, past_value,
            seqlens_k, total_seq_len, cos_cache, sin_cache,
            num_heads=..., kv_num_heads=..., do_rotary=1,
        )

    ``cos_cache`` and ``sin_cache`` are the original rotary embedding
    tables (traced back through the ``Gather`` nodes).
    """

    def __init__(self):
        super().__init__()
        # Cached graph-level values shared across all GQA replacements
        self._seqlens_k = None
        self._total_seq_len = None
        self._cos_cache = None
        self._sin_cache = None

    # ------------------------------------------------------------------ pattern

    def pattern(self, op, q_pre, k_pre, v, attention_bias, past_key, past_value, cos, sin):
        q_rot = op.RotaryEmbedding(
            q_pre,
            cos,
            sin,
            _allow_other_attributes=True,
        )
        k_rot = op.RotaryEmbedding(
            k_pre,
            cos,
            sin,
            _allow_other_attributes=True,
        )
        return op.Attention(
            q_rot,
            k_rot,
            v,
            attention_bias,
            past_key,
            past_value,
            _allow_other_attributes=True,
            _outputs=["attn_out", "present_key", "present_value"],
        )

    # ------------------------------------------------------------------ check

    def check(self, context, attn_out, cos, sin, past_key, past_value, **_):
        result = MatchResult()

        attn = attn_out.producer()
        if attn.attributes.get_float("scale", None) is None:
            return result.fail("Missing scale attribute on Attention")
        if attn.attributes.get_int("q_num_heads", None) is None:
            return result.fail("Missing q_num_heads on Attention")
        if attn.attributes.get_int("kv_num_heads", None) is None:
            return result.fail("Missing kv_num_heads on Attention")

        # cos/sin must come from Gather (position-indexed cache tables)
        cos_prod = cos.producer()
        sin_prod = sin.producer()
        if cos_prod is None or cos_prod.op_type != "Gather":
            return result.fail("cos must be Gather-produced")
        if sin_prod is None or sin_prod.op_type != "Gather":
            return result.fail("sin must be Gather-produced")

        # past_key/past_value must be graph inputs (not None, not computed)
        # This distinguishes decoder attention from vision-encoder attention
        if past_key is None or past_value is None:
            return result.fail("No KV cache inputs")
        if past_key.producer() is not None:
            return result.fail("past_key is not a graph input")
        if past_value.producer() is not None:
            return result.fail("past_value is not a graph input")

        return result

    # ------------------------------------------------------------------ rewrite

    def rewrite(
        self,
        op,
        q_pre,
        k_pre,
        v,
        attention_bias,
        past_key,
        past_value,
        cos,
        sin,
        attn_out,
        present_key,
        present_value,
        **_,
    ):
        attn = attn_out.producer()
        scale = attn.attributes.get_float("scale")
        q_num_heads = attn.attributes.get_int("q_num_heads")
        kv_num_heads = attn.attributes.get_int("kv_num_heads")

        # Trace cos/sin back through Gather to the cache table initializers
        if self._cos_cache is None:
            self._cos_cache = cos.producer().inputs[0]
            self._sin_cache = sin.producer().inputs[0]

        # Build seqlens_k and total_sequence_length once (shared)
        if self._seqlens_k is None:
            graph = attn.graph
            attention_mask = None
            for gi in graph.inputs:
                if gi.name == "attention_mask":
                    attention_mask = gi
                    break

            # seqlens_k = Cast(ReduceSum(attention_mask, axis=1) - 1, INT32)
            axis = op.Constant(value_ints=[1])
            reduce_sum = op.ReduceSum(attention_mask, axis)
            one = op.Constant(value_ints=[1])
            self._seqlens_k = op.Cast(
                op.Sub(reduce_sum, one),
                to=6,
            )

            # total_seq_len = Cast(Gather(Shape(attention_mask), 1), INT32)
            mask_shape = op.Shape(attention_mask)
            idx_1 = op.Constant(value_int=1)
            self._total_seq_len = op.Cast(
                op.Gather(mask_shape, idx_1),
                to=6,
            )

        # Create GroupQueryAttention (3 outputs match Attention's 3)
        outputs = op.op_multi_out(
            "GroupQueryAttention",
            inputs=[
                q_pre,
                k_pre,
                v,
                past_key,
                past_value,
                self._seqlens_k,
                self._total_seq_len,
                self._cos_cache,
                self._sin_cache,
            ],
            domain="com.microsoft",
            attributes={
                "num_heads": q_num_heads,
                "kv_num_heads": kv_num_heads,
                "scale": scale,
                "do_rotary": 1,
                "rotary_interleaved": 0,
            },
            num_outputs=3,
        )

        return outputs[0], outputs[1], outputs[2]


# ====================================================================
# PackQKVForGQA — consolidates 3 separate MatMuls into 1 packed MatMul
# ====================================================================


def _get_weight_tensor(weight: ir.Value) -> np.ndarray:
    """Return a weight as a numpy array with shape ``(out_features, hidden_size)``.

    Handles two patterns:

    1. ``Transpose(constant, perm=[1,0])`` — the constant already has
       shape ``(out_features, hidden_size)``, returned as-is.
    2. Plain constant — shape ``(hidden_size, out_features)``, transposed
       before returning.

    Raises:
        ``MatchFailureError`` if *weight* is not a constant (possibly
        behind a Transpose).
    """
    producer = weight.producer()
    if producer is not None and producer.op_type == "Transpose":
        perm = producer.attributes.get("perm", None)
        if perm is not None and list(perm.value) == [1, 0]:
            tensor = ir.convenience.get_const_tensor(producer.inputs[0])
            if tensor is not None:
                return tensor.numpy()

    tensor = ir.convenience.get_const_tensor(weight)
    if tensor is not None:
        return tensor.numpy().T

    raise MatchFailureError(f"weight {weight.name} is not a constant")


class PackQKVForGQA(RewriteRuleClassBase):
    """Pack separate Q/K/V projections into a single MatMul for GQA.

    This rule runs **after** ``RotaryAttentionToGQA`` and looks for
    ``GroupQueryAttention`` nodes whose Q, K, V inputs each come from a
    separate ``MatMul`` projection that shares the same ``hidden_states``
    input.  The ``fused_matmul`` rewrite must run **after** this rule
    so that projections are still plain ``MatMul`` nodes when this rule
    matches.

    **Matched pattern:**

    .. code-block:: text

        q = MatMul(hidden, W_q)
        k = MatMul(hidden, W_k)
        v = MatMul(hidden, W_v)
        out, pkey, pval = GroupQueryAttention(q, k, v, ...)

    **Replacement:**

    .. code-block:: text

        W_qkv = concatenate([W_q, W_k, W_v])  # normalized to (out, hidden)
        packed = MatMul(hidden, Transpose(W_qkv))
        out, pkey, pval = GroupQueryAttention(packed, None, None, ...)

    Each weight is independently normalized to ``(out_features,
    hidden_size)`` before concatenation, so mixed transpose patterns
    across Q/K/V are handled correctly.  The packed weight is stored as
    a graph initializer (not a ``Constant`` node attribute) so it can
    be serialised to external data files for large models.
    """

    _pack_counter: int
    _qkv_wt_transposed: np.ndarray | None

    def __init__(self):
        super().__init__()
        self._pack_counter = 0
        self._qkv_wt_transposed = None

    # ------------------------------------------------------------------ pattern

    def pattern(self, op, hidden, q_w, k_w, v_w):
        q = op.MatMul(hidden, q_w)
        k = op.MatMul(hidden, k_w)
        v = op.MatMul(hidden, v_w)

        return op.GroupQueryAttention(
            q,
            k,
            v,
            _domain="com.microsoft",
            _allow_other_attributes=True,
            _allow_other_inputs=True,
            _outputs=["gqa_out", "present_key", "present_value"],
        )

    # ------------------------------------------------------------------ check

    def check(self, context, q_w, k_w, v_w, **_):
        # Extract weights normalized to (out_features, hidden_size).
        # Raises MatchFailureError if any weight is not a constant.
        q_np = _get_weight_tensor(q_w)
        k_np = _get_weight_tensor(k_w)
        v_np = _get_weight_tensor(v_w)

        # Concatenate along axis=0: all weights are (out, hidden)
        self._qkv_wt_transposed = np.concatenate([q_np, k_np, v_np], axis=0)
        return True

    # ------------------------------------------------------------------ rewrite

    def rewrite(
        self,
        op,
        hidden,
        gqa_out,
        present_key,
        present_value,
        **_,
    ):
        # Store packed weight as a graph initializer
        self._pack_counter += 1
        w_name = f"packed_qkv_weight_{self._pack_counter}"
        packed_w = op.initializer(ir.Tensor(self._qkv_wt_transposed, name=w_name), name=w_name)
        self._qkv_wt_transposed = None

        # Transpose + MatMul for the packed projection
        packed_w_t = op.Transpose(packed_w, perm=[1, 0])
        packed_qkv = op.MatMul(hidden, packed_w_t)

        # Recover remaining GQA inputs and attributes from the matched node
        gqa_node = gqa_out.producer()
        attrs = {key: gqa_node.attributes[key].value for key in gqa_node.attributes}

        outputs = op.op_multi_out(
            "GroupQueryAttention",
            inputs=[
                packed_qkv,
                None,
                None,
                *gqa_node.inputs[3:],
            ],
            domain="com.microsoft",
            attributes=attrs,
            num_outputs=3,
        )

        return outputs[0], outputs[1], outputs[2]


def group_query_attention_rules() -> RewriteRuleSet:
    """Return rules that fuse RotaryEmbedding + Attention into GQA.

    The rule set contains two rules applied in order:

    1. ``RotaryAttentionToGQA`` -- fuses RotaryEmbedding + Attention into
       ``GroupQueryAttention`` with separate Q, K, V inputs.
    2. ``PackQKVForGQA`` -- consolidates the three separate Q/K/V
       projection MatMuls into a single packed MatMul when possible.

    Returns:
        :class:`RewriteRuleSet` containing both rules.
    """
    return RewriteRuleSet(
        [
            RotaryAttentionToGQA().rule(),
            PackQKVForGQA().rule(),
        ]
    )
