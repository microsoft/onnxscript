# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""LongCat Flash model with dual-sublayer architecture.

Each physical decoder layer contains two MLA attention sub-layers,
two dense MLPs, and one MoE shortcut block computed once per layer.

Reference: LongCat Flash HuggingFace implementation.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    MLP,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class _LongcatFlashMLA(nn.Module):
    """Multi-head Latent Attention for LongCat Flash.

    Extends the standard DeepSeek MLA pattern with LongCat-specific
    LoRA scaling factors applied after the low-rank projections:
    - q_scale = sqrt(hidden_size / q_lora_rank) applied to Q (nope + rope)
    - kv_scale = sqrt(hidden_size / kv_lora_rank) applied to latent KV

    Uses interleaved RoPE (apply_rotary_pos_emb with interleaved=True).

    Weight names match HF:
    - q_a_proj.weight, q_a_layernorm.weight, q_b_proj.weight
    - kv_a_proj_with_mqa.weight, kv_a_layernorm.weight, kv_b_proj.weight
    - o_proj.weight
    """

    def __init__(self, config: ArchitectureConfig, layer_idx: int = 0):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        # Q path: LoRA compression → norm → decompression
        self.q_a_proj = Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # KV path: joint projection → layernorm → decompression
        self.kv_a_proj_with_mqa = Linear(
            config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)

        self.scaling = self.qk_head_dim**-0.5
        # LoRA scaling factors: scale activations after low-rank projections
        # mla_scale_q_lora = sqrt(hidden / q_lora_rank)
        # mla_scale_kv_lora = sqrt(hidden / kv_lora_rank)
        self.mla_scale_q_lora = (config.hidden_size / self.q_lora_rank) ** 0.5
        self.mla_scale_kv_lora = (config.hidden_size / self.kv_lora_rank) ** 0.5

        self._layer_idx = layer_idx

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # --- Q path: LoRA compress → norm → decompress ---
        q_compressed = self.q_a_proj(op, hidden_states)
        q_compressed = self.q_a_layernorm(op, q_compressed)
        q_states = self.q_b_proj(op, q_compressed)

        # Reshape to per-head: (B, S, H * qk_head_dim) → (B, S, H, qk_head_dim)
        q_states = op.Reshape(q_states, [0, 0, self.num_heads, self.qk_head_dim])
        # Split into nope and rope portions: (B, S, H, nope_dim), (B, S, H, rope_dim)
        q_nope, q_rope = op.Split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1, _outputs=2
        )
        # Apply LoRA Q scaling to both nope and rope portions
        q_nope = op.Mul(q_nope, float(self.mla_scale_q_lora))
        q_rope = op.Mul(q_rope, float(self.mla_scale_q_lora))
        # Flatten rope back for RoPE: (B, S, H, rope_dim) → (B, S, H*rope_dim)
        q_rope = op.Reshape(q_rope, [0, 0, -1])

        # --- KV path: joint projection → split → norm → scale ---
        compressed_kv = self.kv_a_proj_with_mqa(op, hidden_states)
        k_pass, k_rope = op.Split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1, _outputs=2
        )
        k_pass = self.kv_a_layernorm(op, k_pass)
        # Apply LoRA KV scaling before decompression
        k_pass = op.Mul(k_pass, float(self.mla_scale_kv_lora))
        # Decompress latent KV → per-head k_nope + v
        kv_decompressed = self.kv_b_proj(op, k_pass)
        kv_decompressed = op.Reshape(
            kv_decompressed,
            [0, 0, self.num_heads, self.qk_nope_head_dim + self.v_head_dim],
        )
        k_nope, value_states = op.Split(
            kv_decompressed, [self.qk_nope_head_dim, self.v_head_dim], axis=-1, _outputs=2
        )
        value_states = op.Reshape(value_states, [0, 0, -1])

        # --- Apply interleaved RoPE ---
        # q_rope: (B, S, H*rope_dim) — apply to all H heads at once
        q_rope = apply_rotary_pos_emb(
            op,
            x=q_rope,
            position_embeddings=position_embeddings,
            num_heads=self.num_heads,
            rotary_embedding_dim=0,
            interleaved=True,
        )
        # k_rope is single-head: (B, S, rope_dim)
        k_rope = apply_rotary_pos_emb(
            op,
            x=k_rope,
            position_embeddings=position_embeddings,
            num_heads=1,
            rotary_embedding_dim=0,
            interleaved=True,
        )
        # Broadcast k_rope to all heads: (B, S, 1, rope_dim) → (B, S, H, rope_dim)
        k_rope_4d = op.Reshape(k_rope, [0, 0, 1, self.qk_rope_head_dim])
        k_rope_expanded = op.Expand(k_rope_4d, [1, 1, self.num_heads, 1])

        # Concatenate nope + rope → final Q and K
        q_rope_4d = op.Reshape(q_rope, [0, 0, self.num_heads, self.qk_rope_head_dim])
        query_4d = op.Concat(q_nope, q_rope_4d, axis=-1)  # (B, S, H, qk_head_dim)
        key_4d = op.Concat(k_nope, k_rope_expanded, axis=-1)  # (B, S, H, qk_head_dim)
        query_states = op.Reshape(query_4d, [0, 0, -1])
        key_states = op.Reshape(key_4d, [0, 0, -1])

        # --- Attention ---
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


class _LongcatFlashRouter(nn.Module):
    """Routing gate for LongCat Flash MoE.

    Computes softmax over all experts, adds correction bias for selection,
    selects top-k via TopK, then uses original (unbiased) softmax scores
    as routing weights scaled by routed_scaling_factor.

    Weight names match HF:
    - classifier.weight: [total_experts, hidden_size]
    - e_score_correction_bias: [total_experts]
    """

    def __init__(self, config: ArchitectureConfig, total_experts: int):
        super().__init__()
        assert config.num_experts_per_tok is not None
        self.classifier = Linear(config.hidden_size, total_experts, bias=False)
        self.e_score_correction_bias = nn.Parameter([total_experts])
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # Compute routing logits: (B*S, total_experts)
        router_logits = self.classifier(op, hidden_states)
        # Softmax over all experts
        router_probs = op.Softmax(router_logits, axis=-1)
        # Add correction bias for expert selection
        biased = op.Add(router_probs, self.e_score_correction_bias)
        # Select top-k experts based on biased scores
        k = op.Constant(value_ints=[self.top_k])
        _, selected_experts = op.TopK(biased, k, axis=-1, _outputs=2)
        # Use original (unbiased) probs as routing weights
        routing_weights = op.GatherElements(router_probs, selected_experts, axis=-1)
        # Scale by routed_scaling_factor
        routing_weights = op.Mul(routing_weights, float(self.routed_scaling_factor))
        return routing_weights, selected_experts


class _LongcatFlashMoE(nn.Module):
    """MoE shortcut block for LongCat Flash.

    Combines n_routed_experts real MLP experts with zero_expert_num identity
    (pass-through) experts. Real experts apply SiLU gate/up/down projections;
    identity experts output the input hidden states unchanged.

    Weight names:
    - router.classifier.weight: [total_experts, hidden_size]
    - router.e_score_correction_bias: [total_experts]
    - experts.{i}.gate_proj.weight, up_proj.weight, down_proj.weight
      (only for i < n_routed_experts; zero experts have no weights)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.moe_intermediate_size is not None
        self.n_routed_experts = config.num_local_experts
        self.zero_expert_num = getattr(config, "zero_expert_num", 0)
        total_experts = self.n_routed_experts + self.zero_expert_num
        self.router = _LongcatFlashRouter(config, total_experts)
        # Only real experts have projection weights
        expert_config = dataclasses.replace(
            config, intermediate_size=config.moe_intermediate_size
        )
        self.experts = nn.ModuleList(
            [MLP(expert_config) for _ in range(self.n_routed_experts)]
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        routing_weights, selected_experts = self.router(op, hidden_states)
        # routing_weights: (B*S, top_k), selected_experts: (B*S, top_k)

        result = None
        total_experts = self.n_routed_experts + self.zero_expert_num
        for expert_idx in range(total_experts):
            if expert_idx < self.n_routed_experts:
                # Real expert: apply gate/up/down MLP projection
                expert_output = self.experts[expert_idx](op, hidden_states)
            else:
                # Identity/zero expert: output is the input unchanged
                expert_output = hidden_states

            expert_id = op.Constant(value_int=expert_idx)
            match = op.Equal(selected_experts, expert_id)
            match_float = op.Cast(match, to=1)  # (B*S, top_k) FLOAT
            weighted = op.Mul(routing_weights, match_float)
            weight = op.ReduceSum(weighted, [-1], keepdims=True)  # (B*S, 1)
            contribution = op.Mul(expert_output, weight)
            if result is None:
                result = contribution
            else:
                result = op.Add(result, contribution)

        return result


class LongcatFlashDecoderLayer(nn.Module):
    """Dual sub-layer physical decoder layer for LongCat Flash.

    Architecture per physical layer:
    1. input_layernorm[0] → self_attn[0] (MLA) → residual add
    2. post_attention_layernorm[0] → mlp (MoE shortcut) + mlps[0] (dense) → residual add
    3. input_layernorm[1] → self_attn[1] (MLA) → residual add
    4. post_attention_layernorm[1] → mlps[1] (dense) → residual add + MoE shortcut

    The MoE shortcut is computed once (from post_attention_layernorm[0] output)
    and added at the end of the physical layer alongside dense MLP 1's residual.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = nn.ModuleList([_LongcatFlashMLA(config), _LongcatFlashMLA(config)])
        self.mlps = nn.ModuleList([MLP(config), MLP(config)])
        self.mlp = _LongcatFlashMoE(config)
        self.input_layernorm = nn.ModuleList(
            [
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            ]
        )
        self.post_attention_layernorm = nn.ModuleList(
            [
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            ]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # Unpack the two KV pairs (one per sub-attention)
        past_kv0, past_kv1 = past_key_value if past_key_value is not None else (None, None)

        # --- Sub-attention 0 ---
        residual = hidden_states
        hidden_states = self.input_layernorm[0](op, hidden_states)
        hidden_states, present_kv0 = self.self_attn[0](
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_kv0,
        )
        hidden_states = op.Add(residual, hidden_states)

        # --- Post-attn-norm-0 → MoE shortcut + Dense MLP 0 ---
        # Both MoE shortcut and dense MLP 0 use the same post-norm output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm[0](op, hidden_states)
        # MoE shortcut is computed once and carried forward to the final residual add
        shortcut_moe_out = self.mlp(op, hidden_states)
        # Dense MLP 0 uses the same post-norm output
        hidden_states = self.mlps[0](op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        # --- Sub-attention 1 ---
        residual = hidden_states
        hidden_states = self.input_layernorm[1](op, hidden_states)
        hidden_states, present_kv1 = self.self_attn[1](
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_kv1,
        )
        hidden_states = op.Add(residual, hidden_states)

        # --- Post-attn-norm-1 → Dense MLP 1 → add MoE shortcut ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm[1](op, hidden_states)
        hidden_states = self.mlps[1](op, hidden_states)
        # Final residual: dense_mlp_1 output + residual + shortcut_moe from earlier
        hidden_states = op.Add(op.Add(residual, hidden_states), shortcut_moe_out)

        return hidden_states, (present_kv0, present_kv1)


class LongcatFlashTextModel(nn.Module):
    """Text decoder backbone for LongCat Flash.

    Uses num_hidden_layers = 2 * num_physical_layers (matching HF convention).
    Each physical LongcatFlashDecoderLayer consumes two consecutive KV cache
    slots and returns two KV pairs appended to the flat present_key_values list.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self._dtype = config.dtype
        # Each physical layer handles 2 sub-attentions, so num_physical_layers = num_hidden_layers // 2
        num_physical_layers = config.num_hidden_layers // 2
        self.layers = nn.ModuleList(
            [LongcatFlashDecoderLayer(config) for _ in range(num_physical_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # RoPE applies only to qk_rope_head_dim; use modified head_dim for cos/sin cache
        if config.qk_rope_head_dim is not None and config.qk_rope_head_dim > 0:
            rope_config = dataclasses.replace(config, head_dim=config.qk_rope_head_dim)
        else:
            rope_config = config
        self.rotary_emb = initialize_rope(rope_config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        # past_key_values has 2 * len(self.layers) entries (2 per physical layer)
        num_virtual_layers = 2 * len(self.layers)
        past_kvs = (
            past_key_values if past_key_values is not None else [None] * num_virtual_layers
        )

        present_key_values = []
        for i, layer in enumerate(self.layers):
            layer_past_kv = (past_kvs[2 * i], past_kvs[2 * i + 1])
            hidden_states, (present_kv0, present_kv1) = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=layer_past_kv,
            )
            # Append both KV pairs; total length = 2 * num_physical_layers = num_hidden_layers
            present_key_values.append(present_kv0)
            present_key_values.append(present_kv1)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class LongcatFlashCausalLMModel(CausalLMModel):
    """LongCat Flash causal LM with dual-sublayer MLA + MoE architecture.

    Each physical decoder layer contains two MLA attention sub-layers, two
    dense MLPs, and one MoE shortcut block computed once per layer.

    model_type: longcat_flash
    """

    default_task: str = "text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = LongcatFlashTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Remap HuggingFace weight names to ONNX parameter names.

        Key mappings:
        - self_attn.{0|1}.*, mlps.{0|1}.*, input_layernorm.{0|1}.*,
          post_attention_layernorm.{0|1}.* align via nn.ModuleList naming.
        - mlp.router.classifier.weight and mlp.router.e_score_correction_bias
          align directly (same path structure).
        - MoE expert weights: HF stores fused tensors, we split per-expert:
            mlp.experts.gate_up_proj [total, 2*mid, hidden] →
              mlp.experts.{i}.gate_proj.weight + mlp.experts.{i}.up_proj.weight
            mlp.experts.down_proj [n_routed, hidden, mid] →
              mlp.experts.{i}.down_proj.weight
        """
        renamed: dict[str, torch.Tensor] = {}
        n_routed = self.config.num_local_experts  # real experts with projection weights

        for key, value in state_dict.items():
            # Split fused gate_up_proj into per-expert gate/up weights
            if key.endswith(".mlp.experts.gate_up_proj"):
                prefix = key[: -len(".mlp.experts.gate_up_proj")]
                mid = value.shape[1] // 2  # expert_ffn_hidden_size
                for i in range(n_routed):
                    renamed[f"{prefix}.mlp.experts.{i}.gate_proj.weight"] = value[i, :mid]
                    renamed[f"{prefix}.mlp.experts.{i}.up_proj.weight"] = value[i, mid:]
                # Zero experts have no weights in our module
                continue

            # Split per-expert down_proj weights (only n_routed real experts)
            if key.endswith(".mlp.experts.down_proj"):
                prefix = key[: -len(".mlp.experts.down_proj")]
                for i in range(value.shape[0]):
                    renamed[f"{prefix}.mlp.experts.{i}.down_proj.weight"] = value[i]
                continue

            renamed[key] = value

        return super().preprocess_weights(renamed)
