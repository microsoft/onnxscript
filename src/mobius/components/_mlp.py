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

    Three-matrix architecture: ``gate_proj → activation → elementwise mul
    with up_proj → down_proj``.  Used by Llama, Qwen, Mistral, etc.

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


class FCMLP(nn.Module):
    """Two-layer fully-connected MLP: ``up_proj → activation → down_proj``.

    Used by models that do NOT use gated linear units (ViT, CLIP, GPT-2,
    Falcon, DistilBERT, Wav2Vec2, BART, Whisper, Nemotron, etc.).

    Default parameter names are ``up_proj`` / ``down_proj``.  Models with
    different HuggingFace weight names (e.g. ``fc1``/``fc2``,
    ``c_fc``/``c_proj``) should rename in ``preprocess_weights()``.

    Note: ``up_proj``/``down_proj`` was chosen over ``fc1``/``fc2`` for
    consistency with the gated :class:`MLP` (``gate_proj``/``up_proj``/
    ``down_proj``) and the broader LLM ecosystem (Llama, Qwen, Mistral).
    HuggingFace models use many different names (fc1/fc2, lin1/lin2,
    c_fc/c_proj, dense_h_to_4h/dense_4h_to_h, etc.) so no single choice
    avoids renames for most models — 7 of 10 consolidated models need
    ``preprocess_weights`` renames regardless.

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Hidden dimension of the inner layer.
        activation: Activation function name (e.g. ``"gelu"``,
            ``"quick_gelu"``, ``"relu"``).  Defaults to ``"gelu"``.
        bias: Whether to include bias in both linear layers.
        linear_class: Factory callable for creating linear layers.
            Defaults to ``Linear``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = True,
        linear_class: type | None = None,
    ):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        self.up_proj = linear_class(hidden_size, intermediate_size, bias=bias)
        self.down_proj = linear_class(intermediate_size, hidden_size, bias=bias)
        self.act_fn = get_activation(activation)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.up_proj(op, x)
        x = self.act_fn(op, x)
        return self.down_proj(op, x)
