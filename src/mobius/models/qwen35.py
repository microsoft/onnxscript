# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

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
from mobius.components._moe import TopKGate
from mobius.components._rms_norm import OffsetRMSNorm
from mobius.components._rotary_embedding import initialize_rope
from mobius.models.base import CausalLMModel
from mobius.models.qwen_vl import (
    Qwen3VLEmbeddingModel,
    Qwen3VLVisionEncoderModel,
    _QwenVLTextMixin,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Qwen3.5 — hybrid linear/full attention
# ---------------------------------------------------------------------------


class Qwen35DecoderLayer(nn.Module):
    """Qwen3.5 decoder layer with hybrid attention.

    Each layer is either ``"linear_attention"`` (GatedDeltaNet) or
    ``"full_attention"`` (Qwen35Attention with output gating), controlled
    by ``config.layer_types[layer_idx]``.

    Both variants use :class:`OffsetRMSNorm` (the *1 + weight* variant)
    for pre-attention and post-attention normalization.
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

        self.mlp = MLP(config)
        self.input_layernorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
        past_key_value: tuple[ir.Value, ir.Value] | None,
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


class Qwen35TextModel(nn.Module):
    """Qwen3.5 text model with hybrid linear/full attention layers.

    Uses :class:`OffsetRMSNorm` for the final norm and creates
    :class:`Qwen35DecoderLayer` instances that dispatch to either
    ``GatedDeltaNet`` or ``Qwen35Attention`` based on
    ``config.layer_types``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        # Embed tokens: (batch, seq_len) → (batch, seq_len, hidden_size)
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        # Compute (cos, sin) for RoPE: each (batch, seq_len, rotary_dim)
        position_embeddings = self.rotary_emb(op, position_ids)

        # Causal attention mask: (batch, 1, seq_len, total_seq_len)
        attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states if input_ids is None else input_ids,
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


class Qwen35CausalLMModel(CausalLMModel):
    """Qwen3.5 causal language model with hybrid linear/full attention.

    Combines standard GQA layers (with output gating) and GatedDeltaNet
    linear attention layers in a single decoder stack.  The per-layer
    attention type is controlled by ``config.layer_types``.

    Full attention layers use standard KV cache. DeltaNet layers carry
    ``conv_state`` and ``recurrent_state`` tensors, managed by
    :class:`HybridCausalLMTask`.
    """

    default_task: str = "hybrid-text-generation"

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = Qwen35TextModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess HuggingFace state dict for Qwen3.5.

        Handles:
        - Stripping ``language_model.`` prefix from HF checkpoint keys
          (HF stores weights as ``model.language_model.*`` in safetensors)
        - Dropping visual encoder keys (``model.visual.*``)
        - Dropping multi-token prediction (MTP) keys (``mtp*``):
          MTP heads are auxiliary decoding heads used only during
          HuggingFace training; they are not needed for inference.
        - Weight tying (``tie_word_embeddings``)
        """
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(("mtp_", "mtp.")):
                continue

            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]

            if stripped.startswith("visual."):
                continue
            if stripped.startswith("language_model."):
                stripped = stripped[len("language_model.") :]
                cleaned[f"model.{stripped}"] = value
            else:
                cleaned[key] = value

        return super().preprocess_weights(cleaned)


class Qwen35MoEBlock(nn.Module):
    """Qwen3.5-MoE sparse MoE block with shared expert.

    Combines top-k expert routing with a shared expert gated by sigmoid.
    Weight names are aligned to the HuggingFace naming convention::

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

        self.gate = TopKGate(config.hidden_size, num_experts, top_k)

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
        routing_weights, selected_experts = self.gate(op, hidden_states)

        # Loop-over-experts dispatch (same pattern as MoELayer)
        result = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(op, hidden_states)
            expert_id = op.Constant(value_int=expert_idx)
            match = op.Equal(selected_experts, expert_id)
            match_float = op.Cast(match, to=1)  # FLOAT
            weighted = op.Mul(routing_weights, match_float)
            weight = op.ReduceSum(weighted, [-1], keepdims=True)
            contribution = op.Mul(expert_output, weight)
            if result is None:
                result = contribution
            else:
                result = op.Add(result, contribution)

        # Shared expert with sigmoid gating
        shared_output = self.shared_expert(op, hidden_states)
        shared_gate = self.shared_expert_gate(op, hidden_states)
        shared_gate = op.Sigmoid(shared_gate)
        shared_output = op.Mul(shared_output, shared_gate)

        result = op.Add(result, shared_output)
        return result


class Qwen35MoEDecoderLayer(Qwen35DecoderLayer):
    """Qwen3.5-MoE decoder layer with hybrid attention and MoE FFN.

    Same hybrid DeltaNet/full-attention architecture as
    :class:`Qwen35DecoderLayer`, but replaces the dense MLP with a
    :class:`Qwen35MoEBlock`.
    """

    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = Qwen35MoEBlock(config)


class Qwen35MoETextModel(nn.Module):
    """Qwen3.5-MoE text backbone (no LM head).

    Same hybrid DeltaNet/full-attention layer structure as
    :class:`Qwen35TextModel`, but each layer uses MoE FFN
    (:class:`Qwen35MoEBlock`) instead of dense MLP.

    HuggingFace class: ``Qwen3_5MoeModel``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen35MoEDecoderLayer(config, layer_idx=i)
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
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

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


class Qwen35MoECausalLMModel(CausalLMModel):
    """Qwen3.5-MoE causal language model.

    Combines the hybrid DeltaNet/full-attention architecture of Qwen3.5
    with Mixture-of-Experts FFN layers that include a shared expert
    gated by sigmoid.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = Qwen35MoETextModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess HuggingFace state dict for Qwen3.5-MoE.

        Handles:
        - Dropping multi-token prediction (MTP) keys (``mtp_*``):
          MTP heads are auxiliary decoding heads used only during
          HuggingFace training; they are not needed for inference.
        - Weight tying (``tie_word_embeddings``)
        - Unpacking fused expert weights (``experts.gate_up_proj``,
          ``experts.down_proj``) into per-expert tensors
        """
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("mtp_"):
                continue

            # Unpack fused expert weights into per-expert tensors.
            # HF format: [num_experts, fused_dim, hidden] with gate+up fused
            if key.endswith(".mlp.experts.gate_up_proj"):
                prefix = key[: -len("experts.gate_up_proj")]
                num_experts = value.shape[0]
                half = value.shape[1] // 2
                for i in range(num_experts):
                    cleaned[f"{prefix}experts.{i}.gate_proj.weight"] = value[i, :half]
                    cleaned[f"{prefix}experts.{i}.up_proj.weight"] = value[i, half:]
                continue

            # Stacked expert down_proj without .weight suffix (HF format)
            if key.endswith(".mlp.experts.down_proj"):
                prefix = key[: -len("experts.down_proj")]
                num_experts = value.shape[0]
                for i in range(num_experts):
                    cleaned[f"{prefix}experts.{i}.down_proj.weight"] = value[i]
                continue

            cleaned[key] = value

        return super().preprocess_weights(cleaned)


# ---------------------------------------------------------------------------
# Qwen3.5-VL — vision-language (3-model split)
# ---------------------------------------------------------------------------


class Qwen35VLTextModel(_QwenVLTextMixin, Qwen35CausalLMModel):
    """Qwen3.5-VL text-only decoder.

    Extracts the text backbone from the Qwen3.5-VL multimodal model.
    Strips ``language_model.`` weight prefixes and drops ``visual.`` keys.
    """


class Qwen35VL3ModelCausalLMModel(nn.Module):
    """Qwen3.5-VL vision-language model (3-model split).

    Builds three separate ONNX models for onnxruntime-genai:

    - ``decoder``: text decoder taking ``inputs_embeds`` (interleaved MRoPE)
    - ``vision_encoder``: packed-attention ViT outputting merged features
    - ``embedding``: token embedding + image feature fusion

    The vision encoder is identical to Qwen3-VL's
    :class:`Qwen3VLVisionModel`.  The text decoder uses
    :class:`Qwen35TextModel` (hybrid linear/full attention).
    """

    default_task: str = "hybrid-qwen-vl"
    category: str = "Multimodal"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = Qwen35VLDecoderModel(config)
        self.vision_encoder = Qwen3VLVisionEncoderModel(config)
        self.embedding = Qwen3VLEmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Qwen35VL3ModelCausalLMModel uses QwenVLTask "
            "which calls each sub-module separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the correct sub-model ONNX initializer names.

        HF keys: ``model.visual.*``, ``model.language_model.*``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Drop multi-token prediction (MTP) keys: MTP heads are
            # auxiliary decoding heads used only during HuggingFace
            # training; they are not needed for inference.
            if key.startswith("mtp_"):
                continue

            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]

            if stripped.startswith("visual."):
                renamed[f"vision_encoder.{stripped}"] = value
            elif stripped.startswith("language_model.embed_tokens."):
                suffix = stripped[len("language_model.") :]
                renamed[f"decoder.model.{suffix}"] = value
                renamed[f"embedding.{suffix}"] = value
                if (
                    self.config.tie_word_embeddings
                    and stripped == "language_model.embed_tokens.weight"
                ):
                    renamed["decoder.lm_head.weight"] = value
            elif stripped.startswith("language_model.lm_head."):
                renamed[f"decoder.{stripped[len('language_model.') :]}"] = value
            elif stripped.startswith("lm_head."):
                renamed[f"decoder.{stripped}"] = value
            elif stripped.startswith("language_model."):
                suffix = stripped[len("language_model.") :]
                renamed[f"decoder.model.{suffix}"] = value
        return renamed


class Qwen35VLDecoderModel(nn.Module):
    """Qwen3.5-VL text decoder taking ``inputs_embeds`` (3-model split).

    Uses interleaved MRoPE with 3D ``position_ids`` of shape
    ``(3, batch, seq_len)`` and hybrid linear/full attention layers.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = Qwen35TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route language_model weights for standalone decoder build."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Drop MTP heads (training-only auxiliary decoders)
            if key.startswith("mtp_"):
                continue
            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]
            if stripped.startswith("visual."):
                continue
            if stripped.startswith("language_model."):
                stripped = stripped[len("language_model.") :]
            renamed[stripped] = value

        if self.config.tie_word_embeddings:
            if "lm_head.weight" not in renamed and "model.embed_tokens.weight" in renamed:
                renamed["lm_head.weight"] = renamed["model.embed_tokens.weight"]
        return renamed
