# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.base import CausalLMModel


class DiffLlamaAttention(nn.Module):
    """Differential Attention for DiffLlama.

    Implements the differential attention mechanism from the DiffLlama paper:
      - Two groups of query heads (split from the full head list) attend to K/V
      - The same paired-value tensor (doubled head-dim V) is used for both groups
      - Differential output: out1 - lambda_full * out2
      - lambda_full = exp(dot(lq1, lk1)) - exp(dot(lq2, lk2)) + lambda_init
      - lambda_init = 0.8 - 0.6 * exp(-0.3 * layer_idx) (constant per layer)
      - Final normalization: (1 - lambda_init) * GroupNorm(diff_out)
        where GroupNorm is RMSNorm without learnable affine parameters

    KV cache format matches CausalLMTask convention:
      past_key:   [B, Hkv, past_seq, D]  (rank 4)
      present_key: [B, Hkv, total_seq, D]  (rank 4)

    Weight names match HuggingFace DiffLlamaAttention exactly:
      self_attn.q_proj, k_proj, v_proj, o_proj (standard projections)
      self_attn.lambda_q1, lambda_k1, lambda_q2, lambda_k2 ([head_dim])
    """

    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__()
        self._nh = config.num_attention_heads
        self._nkv = config.num_key_value_heads
        self._grp = config.num_attention_heads // config.num_key_value_heads
        self._d = config.head_dim
        self._scaling = config.head_dim**-0.5
        self._rope_interleave = config.rope_interleave
        self._rms_norm_eps = config.rms_norm_eps

        # lambda_init is a per-layer constant: 0.8 - 0.6 * exp(-0.3 * layer_idx)
        self._lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)

        self.q_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.k_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.v_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.o_proj = Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attn_o_bias,
        )

        # Learnable lambda parameters (shape: [head_dim])
        self.lambda_q1 = nn.Parameter([config.head_dim])
        self.lambda_k1 = nn.Parameter([config.head_dim])
        self.lambda_q2 = nn.Parameter([config.head_dim])
        self.lambda_k2 = nn.Parameter([config.head_dim])

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
        static_cache=None,
    ):
        """Differential attention forward pass.

        All K/V tensors are kept in rank-4 format [B, Hkv, S, D] to match
        the CausalLMTask KV-cache convention (past_key: [B, Hkv, past_seq, D]).

        Shapes:
          hidden_states → [B, S, H*D]
          q_4d → [B, H, S, D]
          k_4d / v_4d (after GQA expand) → [B, H, Sk, D]
          v_doubled → [B, H, Sk, 2D]  (paired values)
          attn_out → [B, H, S, 2D]
          diff → [B, H/2, S, 2D]  (differential subtraction)
          final → [B, S, H*D]  → o_proj → [B, S, hidden_size]
        """
        nh, nkv, grp, d = self._nh, self._nkv, self._grp, self._d

        q = self.q_proj(op, hidden_states)  # [B, S, H*D]
        k = self.k_proj(op, hidden_states)  # [B, S, Hkv*D]
        v = self.v_proj(op, hidden_states)  # [B, S, Hkv*D]

        # Apply RoPE to Q and K while they are still rank 3
        if position_embeddings is not None:
            q = apply_rotary_pos_emb(
                op,
                x=q,
                position_embeddings=position_embeddings,
                num_heads=nh,
                rotary_embedding_dim=0,
                interleaved=self._rope_interleave,
            )
            k = apply_rotary_pos_emb(
                op,
                x=k,
                position_embeddings=position_embeddings,
                num_heads=nkv,
                rotary_embedding_dim=0,
                interleaved=self._rope_interleave,
            )

        # Dynamic shape scalars
        batch = op.Shape(q, start=0, end=1)  # [B]
        q_len = op.Shape(q, start=1, end=2)  # [S]
        kv_len_cur = op.Shape(k, start=1, end=2)  # [S_new]

        # Reshape K, V to rank 4: [B, Hkv, S_new, D]
        kv_cur_shape = op.Concat(batch, kv_len_cur, [nkv], [d], axis=0)
        k_4d_cur = op.Transpose(op.Reshape(k, kv_cur_shape), perm=[0, 2, 1, 3])
        v_4d_cur = op.Transpose(op.Reshape(v, kv_cur_shape), perm=[0, 2, 1, 3])

        # KV cache concatenation on the sequence axis (axis=2 in rank-4 format)
        # past_key has shape [B, Hkv, past_seq, D] from CausalLMTask convention
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_4d = op.Concat(past_k, k_4d_cur, axis=2)  # [B, Hkv, Sk, D]
            v_4d = op.Concat(past_v, v_4d_cur, axis=2)  # [B, Hkv, Sk, D]
        else:
            k_4d = k_4d_cur  # [B, Hkv, S, D]
            v_4d = v_4d_cur  # [B, Hkv, S, D]

        # Return rank-4 present K/V (matches CausalLMTask output convention)
        present_key = k_4d  # [B, Hkv, Sk, D]
        present_value = v_4d  # [B, Hkv, Sk, D]

        kv_len = op.Shape(k_4d, start=2, end=3)  # [Sk]

        # Reshape Q to rank 4: [B, H, S, D]
        q_shape = op.Concat(batch, q_len, [nh], [d], axis=0)
        q_4d = op.Transpose(op.Reshape(q, q_shape), perm=[0, 2, 1, 3])

        # GQA expand: K, V from [B, Hkv, Sk, D] → [B, H, Sk, D]
        # Unsqueeze to [B, Hkv, 1, Sk, D] → Expand to [B, Hkv, grp, Sk, D]
        # → Reshape to [B, H, Sk, D]
        if grp > 1:
            expand_shape = op.Concat(batch, [nkv], [grp], kv_len, [d], axis=0)
            h_kv_shape = op.Concat(batch, [nh], kv_len, [d], axis=0)
            k_4d = op.Reshape(op.Expand(op.Unsqueeze(k_4d, [2]), expand_shape), h_kv_shape)
            v_4d = op.Reshape(op.Expand(op.Unsqueeze(v_4d, [2]), expand_shape), h_kv_shape)
        # k_4d, v_4d: [B, H, Sk, D]

        # Special V preparation for differential attention:
        #   Split V by first/last H/2 heads, pair head-dims, then repeat heads.
        #   Result: each original head is replaced by [first_half_head || second_half_head]
        # Before: [B, H, Sk, D] — after: [B, H, Sk, 2D]
        v_first, v_second = op.Split(v_4d, [nh // 2, nh // 2], axis=1, _outputs=2)
        paired_v = op.Concat(v_first, v_second, axis=-1)  # [B, H/2, Sk, 2D]
        v_doubled = op.Concat(paired_v, paired_v, axis=1)  # [B, H, Sk, 2D]

        # Scaled dot-product: Q_4d @ K_4d^T / sqrt(D) → [B, H, S, Sk]
        k_t = op.Transpose(k_4d, perm=[0, 1, 3, 2])  # [B, H, D, Sk]
        scale = op.Constant(value_float=float(self._scaling))
        attn_scores = op.Mul(op.MatMul(q_4d, k_t), scale)

        # Causal mask: build [Sk, Sk] lower-triangle via Trilu, then slice
        # last S rows → [S, Sk].  This handles prefix-cache (Sk > S) correctly.
        sk_sq_ones = op.ConstantOfShape(
            op.Concat(kv_len, kv_len, axis=0),
            value=ir.tensor(np.ones(1, dtype=np.float32)),
        )
        lower_tri = op.Trilu(sk_sq_ones, upper=0)  # [Sk, Sk] lower triangle
        past_len = op.Sub(kv_len, q_len)  # [1]
        causal_slice = op.Slice(lower_tri, past_len, kv_len, [0])  # [S, Sk]
        causal_bool = op.Cast(causal_slice, to=9)  # BOOL (dtype 9)
        zero_f = op.Constant(value_float=0.0)
        neg_inf_f = op.Constant(value_float=float("-inf"))
        causal_bias = op.Where(causal_bool, zero_f, neg_inf_f)  # [S, Sk]
        # Unsqueeze to [1, 1, S, Sk] and add to scores [B, H, S, Sk]
        attn_scores = op.Add(attn_scores, op.Unsqueeze(causal_bias, [0, 1]))

        # Padding mask (3D bool [B, S, Sk]) → additive bias [B, 1, S, Sk]
        if attention_bias is not None:
            zero_f2 = op.Constant(value_float=0.0)
            neg_inf_f2 = op.Constant(value_float=float("-inf"))
            pad_bias = op.Where(attention_bias, zero_f2, neg_inf_f2)
            attn_scores = op.Add(attn_scores, op.Unsqueeze(pad_bias, [1]))

        # Softmax and weighted sum with doubled V
        attn_weights = op.Softmax(attn_scores, axis=-1)  # [B, H, S, Sk]
        attn_out = op.MatMul(attn_weights, v_doubled)  # [B, H, S, 2D]

        # Split into two H/2-head groups for differential subtraction
        out1, out2 = op.Split(attn_out, [nh // 2, nh // 2], axis=1, _outputs=2)
        # out1, out2: [B, H/2, S, 2D]

        # Compute lambda_full (scalar):
        #   lambda_full = exp(lq1·lk1) - exp(lq2·lk2) + lambda_init
        lam1 = op.Exp(op.ReduceSum(op.Mul(self.lambda_q1, self.lambda_k1), keepdims=False))
        lam2 = op.Exp(op.ReduceSum(op.Mul(self.lambda_q2, self.lambda_k2), keepdims=False))
        lam_init = op.Constant(value_float=float(self._lambda_init))
        lam_full = op.Add(op.Sub(lam1, lam2), lam_init)

        # Differential output: out1 - lambda_full * out2  →  [B, H/2, S, 2D]
        diff = op.Sub(out1, op.Mul(lam_full, out2))

        # GroupNorm = RMSNorm without learnable affine params (on last dim = 2D)
        #   rms_norm(x) = x / sqrt(mean(x²) + eps)
        x_sq = op.Mul(diff, diff)
        eps = op.Constant(value_float=float(self._rms_norm_eps))
        rms = op.Sqrt(op.Add(op.ReduceMean(x_sq, [-1], keepdims=True), eps))
        scale_norm = op.Constant(value_float=float(1.0 - self._lambda_init))
        normed = op.Mul(op.Div(diff, rms), scale_norm)

        # Reshape [B, H/2, S, 2D] → [B, S, H*D]
        # Transpose: [B, S, H/2, 2D]  then flatten last two dims
        normed_t = op.Transpose(normed, perm=[0, 2, 1, 3])
        out_flat = op.Reshape(normed_t, op.Concat(batch, q_len, [nh * d], axis=0))

        attn_output = self.o_proj(op, out_flat)
        return attn_output, (present_key, present_value)


class DiffLlamaCausalLMModel(CausalLMModel):
    """DiffLlama: differential-attention variant of Llama.

    Replaces each decoder layer's standard Attention with
    DiffLlamaAttention, which computes a weighted difference between two
    attention streams to reduce attention noise.

    All other components (MLP, LayerNorm, embedding, LM head) are
    identical to the base Llama model.

    Replicates HuggingFace's ``DiffLlamaForCausalLM``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = DiffLlamaAttention(config, layer_idx)
