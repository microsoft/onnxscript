# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import LayerNorm, Linear, get_activation
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class NemotronMLP(nn.Module):
    """Nemotron MLP: up_proj → activation → down_proj (no gating)."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()

        self.up_proj = Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        x = self.up_proj(op, x)
        x = self.act_fn(op, x)
        return self.down_proj(op, x)


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
            layer.mlp = NemotronMLP(config)
            layer.input_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = LayerNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
