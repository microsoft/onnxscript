# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._activations import get_activation
from mobius.components._common import Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class MLP(nn.Module):
    """Feed-forward network with gated linear units (GLU-style).

    Args:
        config: Architecture configuration.
        linear_class: Factory callable ``(in_features, out_features, bias=...)``
            for creating projection layers. Defaults to ``Linear``.
    """

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        self.gate_proj = linear_class(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = linear_class(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = linear_class(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        gate = self.act_fn(op, self.gate_proj(op, x))
        up = self.up_proj(op, x)
        return self.down_proj(op, op.Mul(gate, up))
