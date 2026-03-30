# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Cohere and Cohere2 causal language models.

Cohere uses a parallel transformer decoder: a single LayerNorm feeds both the
attention and MLP branches simultaneously, and their outputs are summed into
the residual:

    residual = x
    normed = LayerNorm(x)
    x = residual + attention(normed) + mlp(normed)

This differs from the standard (sequential pre-norm) pattern where each block
has two separate norms.

Replicates HuggingFace's ``CohereForCausalLM`` and ``Cohere2ForCausalLM``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import MLP, Attention, LayerNorm, LayerNormNoBias, StaticCacheState
from mobius.models.base import LayerNormCausalLMModel, LayerNormTextModel

if TYPE_CHECKING:
    import onnx_ir as ir


class _CohereDecoderLayer(nn.Module):
    """Parallel pre-norm decoder layer used by Cohere.

    A single ``input_layernorm`` feeds both attention and MLP; their outputs
    are summed before the residual addition, matching HF ``CohereDecoderLayer``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        # Single shared norm (HF: input_layernorm only, no bias, no post_attention_layernorm)
        self.input_layernorm = LayerNormNoBias(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, rms_norm_class=LayerNorm)
        self.mlp = MLP(config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | StaticCacheState | None,
    ) -> tuple[ir.Value, tuple]:
        if isinstance(past_key_value, StaticCacheState):
            static_cache = past_key_value
            past_key_value = None
        else:
            static_cache = None

        residual = hidden_states
        # Single norm feeds both branches in parallel.
        normed = self.input_layernorm(op, hidden_states)

        attn_out, present_kv = self.self_attn(
            op,
            normed,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            static_cache=static_cache,
        )
        mlp_out = self.mlp(op, normed)

        # Parallel: residual + attn + mlp (no intermediate residual)
        hidden_states = op.Add(residual, op.Add(attn_out, mlp_out))
        return hidden_states, present_kv


class _CohereTextModel(LayerNormTextModel):
    """Cohere text backbone with parallel pre-norm decoder layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Replace the two-norm decoder layers with single-norm parallel layers.
        self.layers = nn.ModuleList(
            [_CohereDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # Final norm is weight-only (no bias) — override LayerNormTextModel's default.
        self.norm = LayerNormNoBias(config.hidden_size, eps=config.rms_norm_eps)


class CohereCausalLMModel(LayerNormCausalLMModel):
    """Cohere and Cohere2 causal language model.

    Differences from ``CausalLMModel``:
    - Uses ``LayerNorm`` (not RMSNorm) for all normalizations.
    - Parallel pre-norm decoder: a single ``input_layernorm`` is shared by
      both the attention and MLP branches; their outputs are summed together
      before the residual addition.
    - ``logit_scale``: the final logits are multiplied by a scalar from the
      HuggingFace config (default 0.0625 = 1/16).

    Replicates HuggingFace's ``CohereForCausalLM`` and ``Cohere2ForCausalLM``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = _CohereTextModel(config)
        self.logit_scale = config.logit_scale

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        logits, present_key_values = super().forward(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        if not math.isclose(self.logit_scale, 1.0):
            # Scale logits by the model's configured logit_scale scalar
            # (HF default: 0.0625 = 1/16 for all Cohere models).
            # CastLike ensures the constant matches logits dtype (fp16/bf16/fp32).
            logits = op.Mul(
                logits, op.CastLike(op.Constant(value_float=float(self.logit_scale)), logits)
            )
        return logits, present_key_values
