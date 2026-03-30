# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mobius._configs import ArchitectureConfig
from mobius.components import FCMLP, OffsetLayerNorm
from mobius.models.base import CausalLMModel


class NemotronCausalLMModel(CausalLMModel):
    """Nemotron model with offset LayerNorm and non-gated MLP.

    Replicates HuggingFace's ``NemotronForCausalLM``.  Key differences
    from the base ``CausalLMModel``:

    - **NemotronLayerNorm1P**: Full LayerNorm (with bias) using a +1
      weight offset — ``LayerNorm(x, weight + 1, bias, eps)``.
    - **Non-gated MLP**: ``down_proj(act_fn(up_proj(x)))`` instead of
      the GLU-style gated MLP.
    - **relu2 activation**: Squared ReLU ``max(x, 0)²``.
    - **partial_rotary_factor = 0.5**: Only half of head dimensions
      receive RoPE positional encoding.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Replace RMSNorm with OffsetLayerNorm (LayerNorm with +1 weight offset)
        self.model.norm = OffsetLayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        for layer in self.model.layers:
            layer.mlp = FCMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=config.hidden_act,
                bias=config.mlp_bias,
            )
            layer.input_layernorm = OffsetLayerNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            layer.post_attention_layernorm = OffsetLayerNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
