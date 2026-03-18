# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Multi-head Latent Attention (MLA) for DeepSeek-V3.

MLA compresses Q and KV projections through low-rank bottlenecks,
reducing KV cache size while maintaining full attention quality.

Reference: DeepSeek-V2/V3 paper, HuggingFace DeepseekV3Attention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import (
    apply_rotary_pos_emb,
)

if TYPE_CHECKING:
    import onnx_ir as ir


class DeepSeekMLA(nn.Module):
    """Multi-head Latent Attention (MLA) for DeepSeek-V2/V3.

    Key differences from standard GQA:
    - Q path: hidden → q_a_proj (LoRA down) → RMSNorm → q_b_proj (LoRA up)
    - KV path: hidden → kv_a_proj_with_mqa → split(kv_lora_rank, rope_dim)
      → kv_a_layernorm → kv_b_proj → split(k_nope, v)
    - RoPE applied only to the rope portion of Q and K (qk_rope_head_dim)
    - Q/K = concat(nope, rope) with qk_head_dim = nope + rope = 192
    - V has different head dim (v_head_dim=128) than Q/K (qk_head_dim=192)
    - k_rope is single-head (broadcast to all heads)
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        scale: float | None = None,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        # Q path: LoRA compression → norm → decompression
        if self.q_lora_rank is not None and self.q_lora_rank > 0:
            self.q_a_proj = Linear(config.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            # No LoRA — direct projection
            self.q_proj = Linear(
                config.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )

        # KV path: joint projection for latent KV + RoPE key
        # Output: (kv_lora_rank) for latent KV + (qk_rope_head_dim) for k_rope
        self.kv_a_proj_with_mqa = Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # Decompresses latent KV into per-head k_nope + v
        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
        )

        # Attention scale: 1/sqrt(qk_head_dim).
        # YaRN mscale is handled by the cos/sin cache (attention_factor
        # is applied to cos/sin in YarnRope), so no additional scaling
        # is needed here. The cos/sin scaling produces attention_factor^2
        # effect on the attention logits automatically.
        self.scaling = scale if scale is not None else self.qk_head_dim**-0.5

        # Whether RoPE uses interleaved layout (DeepSeek-V3 uses this)
        self._rope_interleave = config.rope_interleave

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # --- Q path ---
        if self.q_lora_rank is not None and self.q_lora_rank > 0:
            # LoRA: hidden → q_a_proj → RMSNorm → q_b_proj
            q_compressed = self.q_a_proj(op, hidden_states)
            q_compressed = self.q_a_layernorm(op, q_compressed)
            q_states = self.q_b_proj(op, q_compressed)
        else:
            q_states = self.q_proj(op, hidden_states)

        # Reshape to per-head: (B, S, num_heads * qk_head_dim) → (B, S, num_heads, qk_head_dim)
        q_states = op.Reshape(q_states, [0, 0, self.num_heads, self.qk_head_dim])
        # Split into nope and rope portions: (B, S, H, nope_dim), (B, S, H, rope_dim)
        q_nope, q_rope = op.Split(
            q_states,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            axis=-1,
            _outputs=2,
        )
        # Flatten rope back for RoPE: (B, S, H, rope_dim) → (B, S, H*rope_dim)
        q_rope = op.Reshape(q_rope, [0, 0, -1])

        # --- KV path ---
        # Joint projection: (B, S, hidden) → (B, S, kv_lora_rank + qk_rope_head_dim)
        compressed_kv = self.kv_a_proj_with_mqa(op, hidden_states)

        # Split into latent KV and rope key
        k_pass, k_rope = op.Split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
            _outputs=2,
        )

        # Decompress latent KV → per-head k_nope + v
        k_pass = self.kv_a_layernorm(op, k_pass)
        kv_decompressed = self.kv_b_proj(op, k_pass)
        # (B, S, num_heads * (nope + v_dim)) → (B, S, num_heads, nope + v_dim)
        kv_decompressed = op.Reshape(
            kv_decompressed,
            [0, 0, self.num_heads, self.qk_nope_head_dim + self.v_head_dim],
        )
        k_nope, value_states = op.Split(
            kv_decompressed,
            [self.qk_nope_head_dim, self.v_head_dim],
            axis=-1,
            _outputs=2,
        )
        # k_nope: (B, S, H, nope_dim) → (B, S, H*nope_dim)... not needed yet
        # value_states: (B, S, H, v_dim) → (B, S, H*v_dim) for Attention op
        value_states = op.Reshape(value_states, [0, 0, -1])

        # --- Apply RoPE ---
        # k_rope is single-head: (B, S, rope_dim) — broadcast to all Q heads
        # q_rope: (B, S, H * rope_dim)
        q_rope = apply_rotary_pos_emb(
            op,
            x=q_rope,
            position_embeddings=position_embeddings,
            num_heads=self.num_heads,
            rotary_embedding_dim=0,  # Apply to entire q_rope
            interleaved=self._rope_interleave,
        )
        # k_rope: single head (B, S, rope_dim), apply as 1-head
        k_rope = apply_rotary_pos_emb(
            op,
            x=k_rope,
            position_embeddings=position_embeddings,
            num_heads=1,
            rotary_embedding_dim=0,
            interleaved=self._rope_interleave,
        )
        # Broadcast k_rope to all heads:
        # (B, S, 1, rope_dim) → (B, S, H, rope_dim)
        k_rope_4d = op.Reshape(k_rope, [0, 0, 1, self.qk_rope_head_dim])
        k_rope_expanded = op.Expand(k_rope_4d, [1, 1, self.num_heads, 1])

        # --- Concatenate nope + rope → final Q, K ---
        # q_rope: (B, S, H * rope_dim) → (B, S, H, rope_dim)
        q_rope_4d = op.Reshape(q_rope, [0, 0, self.num_heads, self.qk_rope_head_dim])
        # Q = concat(q_nope, q_rope): (B, S, H, qk_head_dim)
        query_4d = op.Concat(q_nope, q_rope_4d, axis=-1)
        # K = concat(k_nope, k_rope): (B, S, H, qk_head_dim)
        key_4d = op.Concat(k_nope, k_rope_expanded, axis=-1)

        # Flatten back to 3D for Attention op
        query_states = op.Reshape(query_4d, [0, 0, -1])
        key_states = op.Reshape(key_4d, [0, 0, -1])

        # --- Attention ---
        # Note: qk_head_dim (192) != v_head_dim (128).
        # The ONNX Attention op supports different Q/K head dim vs V head dim
        # via the q_num_heads, kv_num_heads, and the actual tensor shapes.
        attn_output, present_key, present_value = op.Attention(
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            kv_num_heads=self.num_heads,
            q_num_heads=self.num_heads,
            scale=self.scaling,
            _outputs=3,
        )

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)
