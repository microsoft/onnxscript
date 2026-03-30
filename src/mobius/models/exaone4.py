# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ExaOne4 causal language model.

Uses post-norm residual connections (norm applied AFTER attention/MLP
sub-layer outputs) and per-head QK RMSNorm, matching
HuggingFace ``Exaone4ForCausalLM``.
"""

from __future__ import annotations

import dataclasses

from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius.components import DecoderLayer
from mobius.models.base import CausalLMModel


class ExaOne4CausalLMModel(CausalLMModel):
    """ExaOne4 model with post-norm ordering and QK-norm.

    Layer pattern: ``hidden → attn → post_attention_layernorm → residual
    → mlp → post_feedforward_layernorm → residual``.
    """

    def __init__(self, config: ArchitectureConfig):
        # Enable QK-norm in config before base class builds the graph
        config = dataclasses.replace(config, attn_qk_norm=True)
        super().__init__(config)
        # Replace all layers with post-norm decoder layers
        self.model.layers = nn.ModuleList(
            [DecoderLayer(config, post_norm=True) for _ in range(config.num_hidden_layers)]
        )
