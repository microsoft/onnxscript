# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""BART encoder-decoder model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._activations import ACT2FN
from mobius.components._common import Embedding, LayerNorm, Linear
from mobius.components._encoder_decoder_attention import (
    EncoderDecoderAttention,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# BART Components
# ---------------------------------------------------------------------------


class _BartEncoderBlock(nn.Module):
    """BART encoder block: post-norm self-attention + post-norm FFN."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = EncoderDecoderAttention(config)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc1 = Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._act_fn = ACT2FN[config.hidden_act]

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        residual = hidden_states
        hidden_states, _ = self.self_attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        hidden_states = self.self_attn_layer_norm(op, hidden_states)

        residual = hidden_states
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = self._act_fn(op, hidden_states)
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        hidden_states = self.final_layer_norm(op, hidden_states)

        return hidden_states


class _BartDecoderBlock(nn.Module):
    """BART decoder block: self-attn + cross-attn + FFN, all post-norm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = EncoderDecoderAttention(config, is_causal=True)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.encoder_attn = EncoderDecoderAttention(config)
        self.encoder_attn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc1 = Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        past_key_value: tuple | None = None,
        cross_past_key_value: ir.Value | None = None,
    ):
        # Self-attention (causal)
        residual = hidden_states
        hidden_states, self_kv = self.self_attn(
            op, hidden_states, past_key_value=past_key_value
        )
        hidden_states = op.Add(residual, hidden_states)
        hidden_states = self.self_attn_layer_norm(op, hidden_states)

        # Cross-attention
        residual = hidden_states
        hidden_states, cross_kv = self.encoder_attn(
            op,
            hidden_states,
            key_value_states=encoder_hidden_states,
            past_key_value=cross_past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)
        hidden_states = self.encoder_attn_layer_norm(op, hidden_states)

        # FFN
        residual = hidden_states
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = self._act_fn(op, hidden_states)
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        hidden_states = self.final_layer_norm(op, hidden_states)

        return hidden_states, self_kv, cross_kv


# ---------------------------------------------------------------------------
# BART Encoder and Decoder
# ---------------------------------------------------------------------------


class _BartEncoder(nn.Module):
    """BART encoder with learned positional embeddings."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        # BART position embeddings have offset of 2
        self.embed_positions = Embedding(
            config.max_position_embeddings + 2, config.hidden_size
        )
        self.layernorm_embedding = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [_BartEncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        inputs_embeds = self.embed_tokens(op, input_ids)

        # Position IDs: 0..seq_len-1, shifted by offset 2
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=2),  # start (offset)
            op.Add(seq_len, op.Constant(value_int=2)),  # end
            op.Constant(value_int=1),  # step
        )
        position_ids = op.Cast(position_ids, to=7)
        position_ids = op.Unsqueeze(position_ids, [0])
        position_embeds = self.embed_positions(op, position_ids)

        hidden_states = op.Add(inputs_embeds, position_embeds)
        hidden_states = self.layernorm_embedding(op, hidden_states)

        for layer in self.layers:
            hidden_states = layer(op, hidden_states)

        return hidden_states


class _BartDecoder(nn.Module):
    """BART decoder with learned positional embeddings, cross-attention, and KV cache."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        num_decoder_layers = getattr(config, "num_decoder_layers", config.num_hidden_layers)
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = Embedding(
            config.max_position_embeddings + 2, config.hidden_size
        )
        self.layernorm_embedding = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [_BartDecoderBlock(config) for _ in range(num_decoder_layers)]
        )
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        encoder_hidden_states: ir.Value,
        position_ids: ir.Value | None = None,
        attention_mask: ir.Value | None = None,
        past_key_values: list | None = None,
        cross_past_key_values: ir.Value | None = None,
    ):
        inputs_embeds = self.embed_tokens(op, input_ids)

        # Position IDs with offset 2, accounting for past KV cache length.
        # HF BART: positions = range(past_kv_len, past_kv_len + seq_len) + 2
        if position_ids is None:
            seq_len = op.Shape(input_ids, start=1, end=2)
            if past_key_values is not None:
                # past_key shape: [batch, num_heads, past_seq_len, head_dim]
                past_len = op.Shape(past_key_values[0][0], start=2, end=3)
            else:
                past_len = op.Constant(value_ints=[0])
            start = op.Add(past_len, op.Constant(value_ints=[2]))
            end = op.Add(start, seq_len)
            position_ids = op.Range(start, end, op.Constant(value_ints=[1]))
            position_ids = op.Cast(position_ids, to=7)
            position_ids = op.Unsqueeze(position_ids, [0])

        position_embeds = self.embed_positions(op, position_ids)
        hidden_states = op.Add(inputs_embeds, position_embeds)
        hidden_states = self.layernorm_embedding(op, hidden_states)

        past_kvs = past_key_values or [None] * len(self.layers)
        cross_past_kvs = cross_past_key_values or [None] * len(self.layers)
        present_self_kvs = []
        present_cross_kvs = []

        for layer, past_kv, cross_kv in zip(self.layers, past_kvs, cross_past_kvs):
            hidden_states, self_kv, cross_kv_out = layer(
                op,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=past_kv,
                cross_past_key_value=cross_kv,
            )
            present_self_kvs.append(self_kv)
            present_cross_kvs.append(cross_kv_out)

        logits = self.lm_head(op, hidden_states)
        return logits, present_self_kvs, present_cross_kvs


# ---------------------------------------------------------------------------
# BART Model
# ---------------------------------------------------------------------------


class BartForConditionalGeneration(nn.Module):
    """BART encoder-decoder model for conditional generation.

    Produces a ModelPackage with separate encoder and decoder components.
    """

    default_task = "seq2seq"
    category = "encoder-decoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.encoder = _BartEncoder(config)
        self.decoder = _BartDecoder(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_bart_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor

        # Shared embeddings
        shared = new_state_dict.pop("shared.weight", None)
        if shared is not None:
            if "encoder.embed_tokens.weight" not in new_state_dict:
                new_state_dict["encoder.embed_tokens.weight"] = shared
            if "decoder.embed_tokens.weight" not in new_state_dict:
                new_state_dict["decoder.embed_tokens.weight"] = shared

        # Tied lm_head
        if "decoder.lm_head.weight" not in new_state_dict:
            embed = new_state_dict.get("encoder.embed_tokens.weight")
            if embed is not None:
                new_state_dict["decoder.lm_head.weight"] = embed

        return new_state_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------


_BART_RENAMES = {
    "self_attn.q_proj.": "self_attn.q_proj.",
    "self_attn.k_proj.": "self_attn.k_proj.",
    "self_attn.v_proj.": "self_attn.v_proj.",
    "self_attn.out_proj.": "self_attn.out_proj.",
    "encoder_attn.q_proj.": "encoder_attn.q_proj.",
    "encoder_attn.k_proj.": "encoder_attn.k_proj.",
    "encoder_attn.v_proj.": "encoder_attn.v_proj.",
    "encoder_attn.out_proj.": "encoder_attn.out_proj.",
}


def _rename_bart_weight(name: str) -> str | None:
    """Rename a HF BART weight to our naming convention."""
    # Strip "model." prefix
    if name.startswith("model."):
        name = name[len("model.") :]
    elif name == "lm_head.weight":
        return "decoder.lm_head.weight"

    if name == "shared.weight":
        return "shared.weight"

    for prefix in ("encoder.", "decoder."):
        if not name.startswith(prefix):
            continue

        rest = name[len(prefix) :]

        # Embeddings pass through
        if rest.startswith(("embed_tokens.", "embed_positions.")):
            return name
        if rest.startswith("layernorm_embedding."):
            return name

        # Layer weights: layers.{i}.{component}
        if rest.startswith("layers."):
            parts = rest.split(".", 2)  # layers, idx, remainder
            if len(parts) < 3:
                return None
            layer_idx = parts[1]
            remainder = parts[2]

            # Attention renames (out_proj aligned with component)
            for old, new in _BART_RENAMES.items():
                if remainder.startswith(old):
                    suffix = remainder[len(old) :]
                    return f"{prefix}layers.{layer_idx}.{new}{suffix}"

            # LayerNorm and FFN pass through
            return name

    return None
