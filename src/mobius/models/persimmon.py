# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Persimmon causal LM with fused QKV and QK LayerNorm.

Architecture: standard sequential pre-norm (like Llama) but with:
- Fused QKV projection (split during ``preprocess_weights``)
- Per-head LayerNorm applied to Q and K before position embeddings
- Non-gated two-matrix MLP (FCMLP)
- Full LayerNorm (not RMSNorm) throughout

Persimmon only supports MHA (no GQA), so ``num_key_value_heads`` is
forced to match ``num_attention_heads``.

Replicates HuggingFace's ``PersimmonForCausalLM``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import rename_mlp_projections, split_interleaved_qkv_weights
from mobius.components import (
    FCMLP,
    Embedding,
    LayerNorm,
    Linear,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import Attention
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class _PersimmonDecoderLayer(nn.Module):
    """Persimmon decoder layer: sequential pre-norm with FCMLP.

    Standard sequential pattern (same as Llama):
    1. ``input_layernorm`` → attention → residual add
    2. ``post_attention_layernorm`` → MLP → residual add

    Uses LayerNorm (not RMSNorm) and FCMLP (not gated MLP). Attention
    uses per-head Q/K LayerNorm (``attn_qk_norm=True`` in config).

    Attribute names match HF ``PersimmonDecoderLayer`` naming (``self_attn``).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Attention with per-head QK LayerNorm; rms_norm_class=LayerNorm
        # ensures q_norm/k_norm use LayerNorm (not RMSNorm) to match HF
        self.self_attn = Attention(config, rms_norm_class=LayerNorm)
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act or "relu2",
            bias=config.mlp_bias,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ) -> tuple[ir.Value, tuple]:
        # --- Attention sub-layer ---
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op, hidden_states, attention_bias, position_embeddings, past_key_value
        )
        hidden_states = op.Add(residual, hidden_states)

        # --- MLP sub-layer ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _PersimmonTextModel(nn.Module):
    """Persimmon backbone with RoPE and full LayerNorm.

    Attribute names match HF ``PersimmonModel``:
    - ``embed_tokens`` for the token embedding
    - ``layers`` for the decoder layer list
    - ``final_layernorm`` for the output norm (not ``norm`` as in Llama)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [_PersimmonDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # HF Persimmon names the final norm "final_layernorm" (not "norm")
        self.final_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op, hidden_states, attention_bias, position_embeddings, past_kv
            )
            present_key_values.append(present_kv)

        hidden_states = self.final_layernorm(op, hidden_states)
        return hidden_states, present_key_values


class PersimmonCausalLMModel(CausalLMModel):
    """Persimmon causal language model with FCMLP and QK LayerNorm.

    Differences from the Llama-style ``CausalLMModel``:
    - Sequential architecture with LayerNorm (not RMSNorm)
    - Fused QKV projection (split in ``preprocess_weights``)
    - Per-head QK LayerNorm in attention (``attn_qk_norm=True``)
    - Non-gated FCMLP instead of gated MLP
    - Final norm is ``model.final_layernorm`` (matches HF attribute)

    Persimmon only supports MHA (no GQA), so ``num_key_value_heads`` is
    forced to match ``num_attention_heads``.

    Replicates HuggingFace's ``PersimmonForCausalLM``.
    """

    default_task: str = "text-generation"
    category: str = "Text Generation"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        # Persimmon is MHA only — override KV heads to prevent shape mismatches
        config = dataclasses.replace(config, num_key_value_heads=config.num_attention_heads)
        # Enable per-head QK LayerNorm (qk_layernorm=True in all Persimmon configs)
        config = dataclasses.replace(config, attn_qk_norm=True)
        self.config = config
        self.model = _PersimmonTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF Persimmon weight names to our ONNX attribute names.

        Most paths match directly (model.embed_tokens, model.layers.N.*,
        model.final_layernorm, lm_head). Transforms needed:

        1. Split fused QKV: ``self_attn.query_key_value.*``
           → ``self_attn.{q,k,v}_proj.*``
        2. Output proj rename: ``self_attn.dense.*``
           → ``self_attn.o_proj.*``
        3. QK norm rename: ``self_attn.q_layernorm.*``
           → ``self_attn.q_norm.*`` (our Attention component uses q_norm/k_norm)
        4. QK norm rename: ``self_attn.k_layernorm.*``
           → ``self_attn.k_norm.*``
        5. MLP up rename:   ``mlp.dense_h_to_4h.*`` → ``mlp.up_proj.*``
        6. MLP down rename: ``mlp.dense_4h_to_h.*`` → ``mlp.down_proj.*``
        """
        # Split per-head interleaved QKV: [h0_q, h0_k, h0_v, h1_q, ...]
        state_dict = split_interleaved_qkv_weights(
            state_dict,
            fused_key="self_attn.query_key_value",
            num_heads=self.config.num_attention_heads,
            kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
        )

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            key = key.replace(".self_attn.dense.", ".self_attn.o_proj.")
            key = key.replace(".self_attn.q_layernorm.", ".self_attn.q_norm.")
            key = key.replace(".self_attn.k_layernorm.", ".self_attn.k_norm.")
            key = rename_mlp_projections(key, "dense_h_to_4h", "dense_4h_to_h")
            new_state_dict[key] = value

        return super().preprocess_weights(new_state_dict)
