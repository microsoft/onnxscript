# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mobius._configs import ArchitectureConfig
from mobius.components import FCMLP, LayerNorm
from mobius.models.base import CausalLMModel


class NemotronCausalLMModel(CausalLMModel):
    """Nemotron model with non-simple LayerNorm and custom MLP order.

    Nemotron uses full LayerNorm (not simplified RMS) with +1 offset,
    and a simpler MLP path: up → activation → down (no gating).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Nemotron uses full LayerNorm (with bias), not RMSNorm
        self.model.norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        for layer in self.model.layers:
            layer.mlp = FCMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=config.hidden_act,
                bias=config.mlp_bias,
            )
            layer.input_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = LayerNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
