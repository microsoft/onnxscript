# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3 model with Multi-head Latent Attention (MLA) and MoE.

Reference: DeepSeek-V3 paper, HuggingFace DeepseekV3ForCausalLM.
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
    Attention,
    Embedding,
    Linear,
    MoELayer,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._deepseek_mla import DeepSeekMLA
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class DeepSeekMoEGate(nn.Module):
    """Expert routing gate for DeepSeek-V2/V3 MoE.

    Supports two scoring modes:
    - sigmoid (V3): sigmoid scoring + correction bias + group TopK
    - softmax (V2/V2-Lite): softmax scoring + simple or group-limited TopK

    Selection method is controlled by topk_method config:
    - "greedy": simple TopK (V2-Lite)
    - "group_limited_greedy": group-based selection with softmax (V2)
    - "noaux_tc": sigmoid + correction bias + group TopK (V3)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method

        self.weight = nn.Parameter([self.num_experts, config.hidden_size])
        # Correction bias only used with sigmoid scoring (V3)
        if self.scoring_func == "sigmoid":
            self.e_score_correction_bias = nn.Parameter([self.num_experts])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # Compute routing logits: hidden @ weight^T
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(
            op.Cast(hidden_states, to=1),
            weight_t,  # Cast to float32
        )

        # Score computation depends on scoring function
        if self.scoring_func == "sigmoid":
            scores = op.Sigmoid(router_logits)  # (B*S, num_experts)
            # Add correction bias for expert selection (V3)
            scores_for_choice = op.Add(scores, self.e_score_correction_bias)
        else:
            # Softmax scoring (V2)
            scores = op.Softmax(router_logits, axis=-1)
            scores_for_choice = scores

        # Expert selection: group-based or simple TopK
        if self.n_group > 1 and self.topk_method != "greedy":
            scores_for_choice = self._group_topk_selection(op, scores_for_choice)

        # Select top-k experts
        k_val = op.Constant(value_ints=[self.top_k])
        _, selected_experts = op.TopK(scores_for_choice, k_val, axis=-1, _outputs=2)

        # Gather original scores (without bias) for selected experts
        routing_weights = op.GatherElements(scores, selected_experts, axis=-1)

        # Normalize weights (V3 with norm_topk_prob=True)
        if self.norm_topk_prob:
            weight_sum = op.ReduceSum(routing_weights, [-1], keepdims=True)
            eps = op.Constant(value_float=1e-20)
            routing_weights = op.Div(routing_weights, op.Add(weight_sum, eps))

        # Apply routing scale
        routing_weights = op.Mul(routing_weights, float(self.routed_scaling_factor))

        return routing_weights, selected_experts

    def _group_topk_selection(self, op, scores_for_choice):
        """Group-based expert selection: pick topk_group groups first."""
        experts_per_group = self.num_experts // self.n_group

        # Flatten batch*seq dims: (B, S, n_experts) → (B*S, n_experts)
        # so group reshaping operates on a 2D tensor (token x expert).
        orig_shape = op.Shape(scores_for_choice)
        flat = op.Reshape(scores_for_choice, [-1, self.num_experts])

        # Reshape to groups: (B*S, n_group, experts_per_group)
        scores_grouped = op.Reshape(flat, [0, self.n_group, experts_per_group])
        # Group score = sum of top-2 within each group
        k_two = op.Constant(value_ints=[2])
        group_top2, _ = op.TopK(scores_grouped, k_two, axis=-1, _outputs=2)
        group_scores = op.ReduceSum(group_top2, [-1], keepdims=False)  # (B*S, n_group)

        # Select top groups
        k_groups = op.Constant(value_ints=[self.topk_group])
        _, group_indices = op.TopK(group_scores, k_groups, axis=-1, _outputs=2)

        # Create mask for selected groups
        group_mask = op.OneHot(
            group_indices,
            self.n_group,
            op.Constant(value_floats=[0.0, 1.0]),
            axis=-1,
        )  # (B*S, topk_group, n_group)
        # Reduce to (B*S, n_group) — 1 if group selected
        group_mask = op.ReduceMax(group_mask, [1], keepdims=False)
        # Expand to per-expert: (B*S, n_group, 1) → (B*S, n_group, experts_per_group)
        group_mask_expanded = op.Reshape(group_mask, [0, self.n_group, 1])
        group_mask_expanded = op.Expand(
            group_mask_expanded,
            [1, 1, experts_per_group],
        )
        # Flatten back: (B*S, num_experts)
        expert_mask = op.Reshape(group_mask_expanded, [0, self.num_experts])
        # Zero out non-selected groups, then reshape back to original (B, S, n_experts)
        return op.Reshape(op.Mul(flat, expert_mask), orig_shape)


class DeepSeekMLADecoderLayer(nn.Module):
    """Decoder layer using Multi-head Latent Attention.

    Forward signature matches DecoderLayer for compatibility with TextModel.
    """

    def __init__(self, config: ArchitectureConfig, is_moe: bool = False):
        super().__init__()
        self.self_attn = DeepSeekMLA(config)
        if is_moe:
            gate = DeepSeekMoEGate(config)
            self.mlp = _DeepSeekMoEFFN(config, gate)
        else:
            self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # Self attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _DeepSeekStandardDecoderLayer(nn.Module):
    """Decoder layer using standard attention (no MLA) with optional MoE.

    Used by DeepSeek-V2 models with use_mla=false (e.g. DeepSeek-OCR-2 LLM).
    Forward signature matches DeepSeekMLADecoderLayer for compatibility.
    """

    def __init__(self, config: ArchitectureConfig, is_moe: bool = False):
        super().__init__()
        self.self_attn = Attention(config)
        if is_moe:
            gate = DeepSeekMoEGate(config)
            self.mlp = _DeepSeekMoEFFN(config, gate)
        else:
            self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # Self attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _DeepSeekMoEFFN(nn.Module):
    """MoE FFN with shared expert for DeepSeek-V3.

    Combines routed experts with a shared expert that processes all tokens.
    Output = moe_routed_output + shared_expert_output.
    """

    def __init__(self, config: ArchitectureConfig, gate: nn.Module):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.moe_intermediate_size is not None
        self.moe = MoELayer(config, gate=gate)
        # Shared expert uses moe_intermediate_size * n_shared_experts
        n_shared = config.n_shared_experts or 1
        shared_intermediate = config.moe_intermediate_size * n_shared
        self.shared_experts = _SharedExpertMLP(config, shared_intermediate)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        moe_output = self.moe(op, hidden_states)
        shared_output = self.shared_experts(op, hidden_states)
        return op.Add(moe_output, shared_output)


class _SharedExpertMLP(nn.Module):
    """Shared expert MLP (same architecture as gate/up/down SiLU MLP)."""

    def __init__(self, config: ArchitectureConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        gate_out = self.gate_proj(op, hidden_states)
        # SiLU = x * sigmoid(x)
        gate = op.Mul(gate_out, op.Sigmoid(gate_out))
        up = self.up_proj(op, hidden_states)
        return self.down_proj(op, op.Mul(gate, up))


class DeepSeekV3TextModel(nn.Module):
    """Text model for DeepSeek-V2/V3 with optional MLA and MoE.

    Architecture:
    - Embedding → N layers → RMSNorm
    - First `first_k_dense_replace` layers use standard MLP
    - Remaining layers use MoE FFN (sigmoid/softmax routing, shared expert)
    - When qk_nope_head_dim > 0: uses Multi-head Latent Attention (MLA)
    - When qk_nope_head_dim == 0 or None: uses standard attention (GQA)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self._dtype = config.dtype

        # Detect MLA vs standard attention
        use_mla = config.qk_nope_head_dim is not None and config.qk_nope_head_dim > 0
        LayerClass = DeepSeekMLADecoderLayer if use_mla else _DeepSeekStandardDecoderLayer  # noqa: N806

        # Build layers: dense for first k, MoE for rest
        first_k = config.first_k_dense_replace
        self.layers = nn.ModuleList(
            [
                LayerClass(config, is_moe=(i >= first_k))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # For MLA, RoPE applies only to the qk_rope_head_dim portion of Q and K,
        # not the full head_dim. Create a modified config so the cos/sin cache
        # has the correct dimensionality: (max_pos, qk_rope_head_dim/2).
        if use_mla and config.qk_rope_head_dim is not None and config.qk_rope_head_dim > 0:
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
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states if input_ids is None else input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
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


class DeepSeekV3CausalLMModel(CausalLMModel):
    """DeepSeek-V3 Causal LM with MLA + MoE.

    model_type: deepseek_v3
    """

    default_task: str = "text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = DeepSeekV3TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Remap HuggingFace weight names to ONNX parameter names.

        Key mappings:
        - MLA attention projections align already (q_a_proj, q_b_proj, etc.)
        - MoE gate: mlp.gate.weight → mlp.moe.gate.weight
        - Shared expert weights align already (mlp.shared_experts.* → same)
        - MoE expert weights: HF stores all experts in a single fused tensor
            experts.gate_up_proj: (n_experts, 2*intermediate, hidden)
            experts.down_proj:    (n_experts, hidden, intermediate)
          These are split into per-expert ONNX weights:
            moe.experts.{i}.gate_proj.weight: (intermediate, hidden)
            moe.experts.{i}.up_proj.weight:   (intermediate, hidden)
            moe.experts.{i}.down_proj.weight: (hidden, intermediate)
        """
        renamed = {}
        for key, value in state_dict.items():
            new_key = key

            # Remap MoE layer names: mlp.gate.* → mlp.moe.gate.*
            new_key = new_key.replace(".mlp.gate.", ".mlp.moe.gate.")

            # HF stores all routed experts in fused tensors:
            # layers.N.mlp.experts.gate_up_proj  (n_experts, 2*mid, hidden)
            # layers.N.mlp.experts.down_proj      (n_experts, hidden, mid)
            # Split into per-expert weights for our ModuleList.
            if new_key.endswith(".mlp.experts.gate_up_proj"):
                prefix = new_key[: -len(".mlp.experts.gate_up_proj")]
                mid = value.shape[1] // 2
                for i in range(value.shape[0]):
                    renamed[f"{prefix}.mlp.moe.experts.{i}.gate_proj.weight"] = value[i, :mid]
                    renamed[f"{prefix}.mlp.moe.experts.{i}.up_proj.weight"] = value[i, mid:]
                continue
            if new_key.endswith(".mlp.experts.down_proj"):
                prefix = new_key[: -len(".mlp.experts.down_proj")]
                for i in range(value.shape[0]):
                    renamed[f"{prefix}.mlp.moe.experts.{i}.down_proj.weight"] = value[i]
                continue

            renamed[new_key] = value

        # Handle weight tying
        return super().preprocess_weights(renamed)
