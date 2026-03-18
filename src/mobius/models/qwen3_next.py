# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Coder-Next: Hybrid DeltaNet + Gated Attention with MoE.

Architecture: 48 layers in a repeating pattern:
  3 x (GatedDeltaNet → MoE) + 1 x (GatedAttention → MoE)

Key features:
- All layers use Mixture-of-Experts FFN (512 experts, top-10)
- Shared expert with sigmoid gating
- OffsetRMSNorm (1 + weight variant)
- Partial RoPE (only 25% of head_dim rotated)
- Standard 1D RoPE (no MRoPE)

Reference: https://huggingface.co/Qwen/Qwen3-Coder-Next
HuggingFace class: Qwen3NextForCausalLM
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._attention import Qwen35Attention
from mobius.components._common import (
    Embedding,
    Linear,
    create_attention_bias,
)
from mobius.components._gated_deltanet import GatedDeltaNet
from mobius.components._mlp import MLP
from mobius.components._moe import SoftmaxTopKGate
from mobius.components._rms_norm import OffsetRMSNorm
from mobius.components._rotary_embedding import initialize_rope
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class Qwen3NextMoEBlock(nn.Module):
    """Qwen3-Next sparse MoE block with shared expert.

    Uses softmax-first routing (SoftmaxTopKGate): softmax over all
    experts before selecting top-k, with optional renormalization.
    Includes a shared expert gated by sigmoid.

    Weight names::

        gate.weight            → router logits
        experts.N.{gate,up,down}_proj.weight
        shared_expert.{gate,up,down}_proj.weight
        shared_expert_gate.weight   → sigmoid gate for shared expert
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        num_experts = config.num_local_experts
        top_k = config.num_experts_per_tok

        self.gate = SoftmaxTopKGate(
            config.hidden_size,
            num_experts,
            top_k,
            norm_topk_prob=config.norm_topk_prob,
        )

        expert_config = dataclasses.replace(
            config, intermediate_size=config.moe_intermediate_size
        )
        self.experts = nn.ModuleList([MLP(expert_config) for _ in range(num_experts)])

        shared_config = dataclasses.replace(
            config,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert = MLP(shared_config)
        self.shared_expert_gate = Linear(config.hidden_size, 1, bias=False)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # Route tokens to experts via softmax top-k gating
        # routing_weights: (B, T, top_k) — gating weights per selected expert
        # selected_experts: (B, T, top_k) — indices of selected experts
        routing_weights, selected_experts = self.gate(op, hidden_states)

        # Loop-over-experts dispatch: run each expert on all tokens,
        # then zero out contributions for tokens not routed to it
        result = None
        for expert_idx, expert in enumerate(self.experts):
            # expert_output: (B, T, hidden_size)
            expert_output = expert(op, hidden_states)
            expert_id = op.Constant(value_int=expert_idx)
            # match: (B, T, top_k) — True where this expert was selected
            match = op.Equal(selected_experts, expert_id)
            match_float = op.Cast(match, to=1)  # FLOAT
            # Sum matched routing weights across top_k dimension → (B, T, 1)
            weighted = op.Mul(routing_weights, match_float)
            weight = op.ReduceSum(weighted, [-1], keepdims=False)
            weight = op.Unsqueeze(weight, [-1])
            # Scale expert output by routing weight
            contribution = op.Mul(expert_output, weight)
            if result is None:
                result = contribution
            else:
                result = op.Add(result, contribution)

        # Shared expert with sigmoid gating (always active, not routed)
        shared_output = self.shared_expert(op, hidden_states)
        shared_gate = self.shared_expert_gate(op, hidden_states)
        shared_gate = op.Sigmoid(shared_gate)  # (B, T, 1)
        shared_output = op.Mul(shared_output, shared_gate)

        result = op.Add(result, shared_output)
        return result


class Qwen3NextDecoderLayer(nn.Module):
    """Qwen3-Next decoder layer with hybrid attention and MoE.

    Each layer is either ``"linear_attention"`` (GatedDeltaNet) or
    ``"full_attention"`` (Qwen35Attention with output gating), and all
    layers use MoE FFN.

    Uses :class:`OffsetRMSNorm` (``1 + weight``) for normalization.
    """

    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__()
        layer_types = config.layer_types or []
        self.layer_type: str = (
            layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = Qwen35Attention(config)

        self.mlp = Qwen3NextMoEBlock(config)
        self.input_layernorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        if self.layer_type == "linear_attention":
            # DeltaNet states are passed through past_key_value as
            # (conv_state, recurrent_state), same tuple pattern as KV cache
            conv_state, recurrent_state = past_key_value

            attn_output, new_conv_state, new_recurrent_state = self.linear_attn(
                op, hidden_states, conv_state, recurrent_state
            )
            present_key_value = (new_conv_state, new_recurrent_state)
        else:
            attn_output, present_key_value = self.self_attn(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
            )

        hidden_states = op.Add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_key_value


class Qwen3NextTextModel(nn.Module):
    """Qwen3-Next text backbone (no LM head).

    Stacks ``Qwen3NextDecoderLayer`` instances that alternate between
    ``GatedDeltaNet`` (linear attention) and ``Qwen35Attention`` (full
    attention with output gating) based on the ``layer_types`` config.
    All layers share the same MoE FFN.

    HuggingFace class: ``Qwen3NextModel``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen3NextDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        # input_ids: (batch, seq_len) → hidden_states: (batch, seq_len, hidden)
        hidden_states = self.embed_tokens(op, input_ids)
        # position_embeddings: (cos, sin) each (batch, seq_len, rotary_dim)
        position_embeddings = self.rotary_emb(op, position_ids)

        # Causal attention bias from attention_mask: (batch, 1, seq, total_seq)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values: list = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class Qwen3NextCausalLMModel(CausalLMModel):
    """Qwen3-Coder-Next causal language model.

    Hybrid DeltaNet + Gated Attention architecture with MoE FFN on all
    layers. Uses 3B activated parameters (80B total) with 512 experts,
    top-10 routing, and a shared expert with sigmoid gating.

    HuggingFace class: ``Qwen3NextForCausalLM``
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = Qwen3NextTextModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess HuggingFace state dict for Qwen3-Next.

        Handles:
        - Dropping multi-token prediction (MTP) keys (``mtp_*``)
        - Weight tying (``tie_word_embeddings``)
        - Unpacking fused expert weights (``experts.gate_up_proj``,
          ``experts.down_proj``) into per-expert tensors
        - Reordering fused DeltaNet projections:
          ``in_proj_qkvz`` (grouped layout) → ``in_proj_qkv`` + ``in_proj_z``
          ``in_proj_ba`` (grouped layout) → ``in_proj_b`` + ``in_proj_a``
        """
        num_k_heads = self.config.linear_num_key_heads or 16
        num_v_heads = self.config.linear_num_value_heads or 32
        head_k_dim = self.config.linear_key_head_dim or 128
        head_v_dim = self.config.linear_value_head_dim or 128
        ratio = num_v_heads // num_k_heads  # V/Z heads per K group

        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("mtp_"):
                continue

            # Unpack fused expert weights into per-expert tensors
            if key.endswith(".mlp.experts.gate_up_proj"):
                prefix = key[: -len("experts.gate_up_proj")]
                n_exp = value.shape[0]
                half = value.shape[1] // 2
                for i in range(n_exp):
                    cleaned[f"{prefix}experts.{i}.gate_proj.weight"] = value[i, :half]
                    cleaned[f"{prefix}experts.{i}.up_proj.weight"] = value[i, half:]
                continue
            if key.endswith(".mlp.experts.down_proj"):
                prefix = key[: -len("experts.down_proj")]
                n_exp = value.shape[0]
                for i in range(n_exp):
                    cleaned[f"{prefix}experts.{i}.down_proj.weight"] = value[i]
                continue

            # Reorder fused DeltaNet projections from grouped to flat layout
            if key.endswith(".linear_attn.in_proj_qkvz.weight"):
                prefix = key[: -len("in_proj_qkvz.weight")]
                # Weight shape: (out_features, hidden_size)
                # Out features arranged as num_k_heads groups, each:
                #   [Q: head_k_dim, K: head_k_dim,
                #    V: ratio*head_v_dim, Z: ratio*head_v_dim]
                per_group = 2 * head_k_dim + 2 * ratio * head_v_dim
                w = value.reshape(num_k_heads, per_group, -1)
                q_parts = w[:, :head_k_dim]
                k_parts = w[:, head_k_dim : 2 * head_k_dim]
                v_parts = w[
                    :,
                    2 * head_k_dim : 2 * head_k_dim + ratio * head_v_dim,
                ]
                z_parts = w[:, 2 * head_k_dim + ratio * head_v_dim :]
                # Flatten: (num_k_heads, dim, hidden) → (total_dim, hidden)
                q_flat = q_parts.reshape(-1, value.shape[-1])
                k_flat = k_parts.reshape(-1, value.shape[-1])
                v_flat = v_parts.reshape(-1, value.shape[-1])
                z_flat = z_parts.reshape(-1, value.shape[-1])
                # Combine Q+K+V into in_proj_qkv, Z into in_proj_z
                cleaned[f"{prefix}in_proj_qkv.weight"] = torch.cat(
                    [q_flat, k_flat, v_flat], dim=0
                )
                cleaned[f"{prefix}in_proj_z.weight"] = z_flat
                continue

            if key.endswith(".linear_attn.in_proj_ba.weight"):
                prefix = key[: -len("in_proj_ba.weight")]
                # Weight shape: (out_features, hidden_size)
                # Out features: num_k_heads groups of (2*ratio) rows
                per_group_ba = 2 * ratio
                w = value.reshape(num_k_heads, per_group_ba, -1)
                b_parts = w[:, :ratio]
                a_parts = w[:, ratio:]
                cleaned[f"{prefix}in_proj_b.weight"] = b_parts.reshape(-1, value.shape[-1])
                cleaned[f"{prefix}in_proj_a.weight"] = a_parts.reshape(-1, value.shape[-1])
                continue

            cleaned[key] = value

        return super().preprocess_weights(cleaned)
