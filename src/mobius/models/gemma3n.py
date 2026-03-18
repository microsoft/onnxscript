# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemma3n text model with AltUp, Laurel, and per-layer input gating.

Gemma3n extends Gemma3 with three efficiency features:
1. **AltUp (Alternating Updates)**: Only a subset of hidden dims are updated
   each layer, reducing compute per layer.
2. **Laurel (Low-rank Residual)**: A low-rank residual augmentation added after
   attention normalization.
3. **Per-layer input gating**: Each layer receives a per-layer embedding derived
   from the input tokens, gated and projected into the hidden space.

The base attention and MLP structure is similar to Gemma3 (hybrid global+sliding
window attention, QK-norm, four-norm decoder layers).
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import Gemma3nConfig
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    Linear,
    OffsetRMSNorm,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class Gemma3nScaledWordEmbedding(Embedding):
    """Embedding with scaling by sqrt(hidden_size)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value):
        embeddings = super().forward(op, input_ids)
        return op.Mul(embeddings, self.embed_scale)


class Gemma3nLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer (Laurel).

    Applies a low-rank residual: output = x + RMSNorm(W_right @ W_left @ x).
    """

    def __init__(self, hidden_size: int, laurel_rank: int, eps: float = 1e-6):
        super().__init__()
        self.linear_left = Linear(hidden_size, laurel_rank, bias=False)
        self.linear_right = Linear(laurel_rank, hidden_size, bias=False)
        self.post_laurel_norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        laurel_hidden = self.linear_left(op, hidden_states)
        laurel_hidden = self.linear_right(op, laurel_hidden)
        normed = self.post_laurel_norm(op, laurel_hidden)
        return op.Add(hidden_states, normed)


class Gemma3nAltUp(nn.Module):
    """Alternating Updates (AltUp).

    Wraps transformer layers with predict/correct steps that enable sparse
    dimension updates. Only the active prediction is processed through the
    transformer layer; the rest are corrected using learned coefficients.

    See: https://proceedings.neurips.cc/paper_files/paper/2023/file/
    f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self.altup_num_inputs = config.altup_num_inputs
        self.altup_active_idx = config.altup_active_idx
        self.hidden_size = config.hidden_size

        self.correction_coefs = Linear(
            config.altup_num_inputs, config.altup_num_inputs, bias=False
        )
        self.prediction_coefs = Linear(
            config.altup_num_inputs, config.altup_num_inputs**2, bias=False
        )
        self.modality_router = Linear(config.hidden_size, config.altup_num_inputs, bias=False)
        self.router_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.router_input_scale = float(config.hidden_size**-1.0)

    def _compute_router_modalities(self, op: builder.OpBuilder, x):
        """Compute router modalities: tanh(router(norm(x) * scale))."""
        router_input = self.router_norm(op, x)
        scale = op.Constant(value_float=self.router_input_scale)
        router_input = op.Mul(router_input, scale)
        routed = self.modality_router(op, router_input)
        return op.Tanh(routed)

    def predict(self, op: builder.OpBuilder, hidden_states_list: list):
        """Predict step: modify inputs using learned prediction coefficients.

        Args:
            hidden_states_list: List of altup_num_inputs tensors, each
                [batch, seq_len, hidden_size].

        Returns:
            List of predicted tensors.
        """
        active = hidden_states_list[self.altup_active_idx]
        modalities = self._compute_router_modalities(op, active)

        # prediction_coefs projects modalities to num_inputs^2 coefficients
        all_coefs = self.prediction_coefs(op, modalities)
        # Reshape to [batch, seq, num_inputs, num_inputs]
        all_coefs = op.Reshape(
            all_coefs,
            op.Constant(value_ints=[0, 0, self.altup_num_inputs, self.altup_num_inputs]),
        )
        # Transpose to [batch, seq, num_inputs_out, num_inputs_in]
        all_coefs = op.Transpose(all_coefs, perm=[0, 1, 3, 2])

        # Stack hidden states: [batch, seq, hidden, num_inputs]
        stacked = op.Concat(*[op.Unsqueeze(h, [-1]) for h in hidden_states_list], axis=-1)
        # matmul: [batch, seq, hidden, num_inputs] x [batch, seq, num_inputs, num_inputs]
        # -> [batch, seq, hidden, num_inputs]
        predictions_stacked = op.MatMul(stacked, all_coefs)
        # Add residual
        predictions_stacked = op.Add(predictions_stacked, stacked)

        # Split back to list
        predictions = []
        for i in range(self.altup_num_inputs):
            idx = op.Constant(value_ints=[i])
            pred_i = op.Gather(predictions_stacked, idx, axis=-1)
            pred_i = op.Squeeze(pred_i, [-1])
            predictions.append(pred_i)
        return predictions

    def correct(self, op: builder.OpBuilder, predictions: list, activated):
        """Correct step: propagate transformer output to all predictions.

        Args:
            predictions: List of predicted tensors from predict step.
            activated: Output of the transformer layer for the active prediction.

        Returns:
            List of corrected prediction tensors.
        """
        modalities = self._compute_router_modalities(op, activated)
        innovation = op.Sub(activated, predictions[self.altup_active_idx])

        # correction_coefs: [batch, seq, num_inputs] + 1
        all_coefs = self.correction_coefs(op, modalities)
        all_coefs = op.Add(all_coefs, 1.0)

        corrected = []
        for i in range(self.altup_num_inputs):
            idx = op.Constant(value_ints=[i])
            coef_i = op.Gather(all_coefs, idx, axis=-1)
            scaled_innovation = op.Mul(innovation, coef_i)
            corrected_i = op.Add(predictions[i], scaled_innovation)
            corrected.append(corrected_i)
        return corrected

    def scale_corrected_output(self, op: builder.OpBuilder, corrected, scale):
        """Apply per-dimension scaling to the corrected output."""
        return op.Mul(corrected, scale)


class Gemma3nDecoderLayer(nn.Module):
    """Gemma3n decoder layer with AltUp, Laurel, and per-layer input gating."""

    def __init__(self, config: Gemma3nConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, rms_norm_class=OffsetRMSNorm)
        self.mlp = MLP(config)
        self.input_layernorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.altup = Gemma3nAltUp(config)
        self.laurel = Gemma3nLaurelBlock(
            config.hidden_size, config.laurel_rank, eps=config.rms_norm_eps
        )

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.per_layer_input_gate = Linear(
            config.hidden_size, config.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = Linear(
            config.hidden_size_per_layer_input, config.hidden_size, bias=False
        )
        self.post_per_layer_input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.altup_active_idx = config.altup_active_idx
        self.altup_correct_scale = config.altup_correct_scale

        # Placed on DecoderLayer (not AltUp) so __call__ realizes it
        self.correct_output_scale = nn.Parameter([config.hidden_size])

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states_list: list,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        per_layer_input: ir.Value,
        past_key_value: tuple | None,
    ):
        # AltUp predict
        predictions = self.altup.predict(op, hidden_states_list)
        active = predictions[self.altup_active_idx]

        # Pre-attention norm + Laurel
        active_normed = self.input_layernorm(op, active)
        laurel_output = self.laurel(op, active_normed)

        # Self attention
        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=active_normed,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        attn_output = self.post_attention_layernorm(op, attn_output)

        # Residual + Laurel (with sqrt(2) normalization)
        attn_gated = op.Add(active, attn_output)
        attn_laurel = op.Add(attn_gated, laurel_output)
        attn_laurel = op.Div(attn_laurel, op.Constant(value_float=float(math.sqrt(2))))

        # MLP
        mlp_input = self.pre_feedforward_layernorm(op, attn_laurel)
        mlp_output = self.mlp(op, mlp_input)
        mlp_output = self.post_feedforward_layernorm(op, mlp_output)
        layer_output = op.Add(attn_laurel, mlp_output)

        # AltUp correct
        corrected = self.altup.correct(op, predictions, layer_output)

        # Scale and apply per-layer input
        first = corrected[self.altup_active_idx]
        if self.altup_correct_scale:
            first = self.altup.scale_corrected_output(op, first, self.correct_output_scale)

        gated = self.per_layer_input_gate(op, first)
        gated = op.Mul(gated, per_layer_input)
        projected = self.per_layer_projection(op, gated)
        projected = self.post_per_layer_input_norm(op, projected)

        # Add projected per-layer input to non-active predictions
        for i in range(len(corrected)):
            if i != self.altup_active_idx:
                corrected[i] = op.Add(corrected[i], projected)

        return corrected, present_key_value


class Gemma3nTextModel(nn.Module):
    """Gemma3n text model with AltUp, Laurel, and hybrid attention."""

    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self._dtype = config.dtype

        embed_scale = float(np.float16(config.hidden_size**0.5))
        self.embed_tokens = Gemma3nScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )
        self.layers = nn.ModuleList(
            [Gemma3nDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.layer_types = config.layer_types
        self.sliding_window = config.sliding_window

        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

        # Local RoPE for sliding window layers
        local_config = copy.deepcopy(config)
        local_config.rope_theta = config.rope_local_base_freq
        local_config.rope_type = "default"
        local_config.rope_scaling = None
        self.rotary_emb_local = initialize_rope(local_config)

        # Per-layer input embeddings
        self.embed_tokens_per_layer = Gemma3nScaledWordEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            config.pad_token_id,
            embed_scale=float(config.hidden_size_per_layer_input**0.5),
        )

        self.per_layer_model_projection = Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = RMSNorm(
            config.hidden_size_per_layer_input, eps=config.rms_norm_eps
        )

        # AltUp projections for expanding input embeddings to altup_num_inputs copies
        self.altup_num_inputs = config.altup_num_inputs
        self.altup_active_idx = config.altup_active_idx
        self.altup_projections = nn.ModuleList(
            [
                Linear(config.hidden_size, config.hidden_size, bias=False)
                for _ in range(config.altup_num_inputs - 1)
            ]
        )
        self.altup_unembed_projections = nn.ModuleList(
            [
                Linear(config.hidden_size, config.hidden_size, bias=False)
                for _ in range(config.altup_num_inputs - 1)
            ]
        )

        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.per_layer_projection_scale = float(config.hidden_size**-0.5)
        self.per_layer_input_scale = float(1.0 / math.sqrt(2.0))

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
            hidden_states_0 = inputs_embeds
        else:
            hidden_states_0 = self.embed_tokens(op, input_ids)

        # Compute per-layer inputs
        per_layer_inputs = self._compute_per_layer_inputs(op, input_ids, hidden_states_0)

        position_embeddings_dict = {
            "full_attention": self.rotary_emb(op, position_ids),
            "sliding_attention": self.rotary_emb_local(op, position_ids),
        }

        # Use hidden_states_0 for query length when input_ids is None (VL decoder path)
        query_input = input_ids if input_ids is not None else hidden_states_0
        attention_bias_dict = {
            "full_attention": create_attention_bias(
                op,
                input_ids=query_input,
                attention_mask=attention_mask,
                dtype=self._dtype,
            ),
            "sliding_attention": create_attention_bias(
                op,
                input_ids=query_input,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                dtype=self._dtype,
            ),
        }

        # Expand to AltUp inputs
        hidden_states_list = [hidden_states_0]
        for proj in self.altup_projections:
            altup_proj = proj(op, hidden_states_0)
            hidden_states_list.append(altup_proj)

        # Decoder layers
        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for i, (layer, layer_type, past_kv) in enumerate(
            zip(self.layers, self.layer_types, past_kvs)
        ):
            # Extract per-layer input for this layer
            per_layer_input = self._get_per_layer_input(op, per_layer_inputs, i)

            hidden_states_list, present_kv = layer(
                op,
                hidden_states_list=hidden_states_list,
                attention_bias=attention_bias_dict[layer_type],
                position_embeddings=position_embeddings_dict[layer_type],
                per_layer_input=per_layer_input,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        # Collapse AltUp outputs back to single hidden state
        result_list = [hidden_states_list[0]]
        for i, proj in enumerate(self.altup_unembed_projections):
            unembed = proj(op, hidden_states_list[i + 1])
            result_list.append(unembed)

        # Average all AltUp outputs
        hidden_states = result_list[0]
        for h in result_list[1:]:
            hidden_states = op.Add(hidden_states, h)
        scale = 1.0 / self.altup_num_inputs
        hidden_states = op.Mul(hidden_states, scale)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values

    def _compute_per_layer_inputs(self, op, input_ids, inputs_embeds):
        """Compute per-layer input embeddings from input_ids and model projection."""
        # Per-layer token embeddings
        if input_ids is not None:
            per_layer_token_embed = self.embed_tokens_per_layer(op, input_ids)
        else:
            per_layer_token_embed = None

        # Per-layer projection from hidden states
        per_layer_proj = self.per_layer_model_projection(op, inputs_embeds)
        per_layer_proj = op.Mul(per_layer_proj, self.per_layer_projection_scale)

        # Reshape to [batch, seq, num_layers, per_layer_dim]
        per_layer_proj = op.Reshape(
            per_layer_proj,
            op.Constant(
                value_ints=[0, 0, self.num_hidden_layers, self.hidden_size_per_layer_input]
            ),
        )
        per_layer_proj = self.per_layer_projection_norm(op, per_layer_proj)

        if per_layer_token_embed is not None:
            per_layer_token_embed = op.Reshape(
                per_layer_token_embed,
                op.Constant(
                    value_ints=[0, 0, self.num_hidden_layers, self.hidden_size_per_layer_input]
                ),
            )
            per_layer_inputs = op.Add(per_layer_proj, per_layer_token_embed)
            per_layer_inputs = op.Mul(per_layer_inputs, self.per_layer_input_scale)
        else:
            per_layer_inputs = per_layer_proj

        return per_layer_inputs

    def _get_per_layer_input(self, op, per_layer_inputs, layer_idx: int):
        """Extract the per-layer input for a specific layer."""
        idx = op.Constant(value_ints=[layer_idx])
        # per_layer_inputs shape: [batch, seq, num_layers, per_layer_dim]
        # Gather on axis=2 to get [batch, seq, per_layer_dim]
        result = op.Gather(per_layer_inputs, idx, axis=2)
        return op.Squeeze(result, [2])


class Gemma3nCausalLMModel(CausalLMModel):
    """Gemma3n causal LM with AltUp, Laurel, and hybrid attention.

    Extends CausalLMModel with the Gemma3n text backbone that includes
    alternating updates, learned augmented residuals, and per-layer
    input gating for mobile efficiency.
    """

    config_class: type = Gemma3nConfig

    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        self.model = Gemma3nTextModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess weights, handling language_model prefix from multimodal."""
        for key in list(state_dict.keys()):
            if "language_model." in key:
                new_key = key.replace("language_model.", "")
                state_dict[new_key] = state_dict.pop(key)
            elif "vision_tower" in key or "multi_modal_projector" in key:
                state_dict.pop(key)
            elif "audio_tower" in key:
                state_dict.pop(key)

        # correct_output_scale lives on DecoderLayer, not nested in altup
        for key in list(state_dict.keys()):
            if ".altup.correct_output_scale" in key:
                new_key = key.replace(".altup.correct_output_scale", ".correct_output_scale")
                state_dict[new_key] = state_dict.pop(key)

        return super().preprocess_weights(state_dict)
