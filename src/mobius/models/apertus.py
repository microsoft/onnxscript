# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Apertus causal language model.

Uses FCMLP with learnable xIELU activation, QK-norm, and renamed layer
norms (``attention_layernorm`` / ``feedforward_layernorm``).

Replicates HuggingFace ``ApertusForCausalLM``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class XIELUActivation(nn.Module):
    """xIELU activation with learnable alpha_p and alpha_n parameters.

    For x > 0: softplus(alpha_p) * x² + beta * x
    For x ≤ 0: (expm1(min(x, eps)) - x) * (beta + softplus(alpha_n)) + beta * x

    Parameters alpha_p and alpha_n are stored in Softplus-inverse space.
    """

    def __init__(self, beta: float = 0.5, eps: float = -1e-6):
        super().__init__()
        # All scalars as nn.Parameter to get unique per-instance names.
        # Shape [1]; values loaded from HF weights via preprocess_weights.
        self.alpha_p = nn.Parameter([1])
        self.alpha_n = nn.Parameter([1])
        self.beta = nn.Parameter([1])
        self.eps = nn.Parameter([1])

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # Learnable activation scales
        alpha_p_act = op.Softplus(self.alpha_p)
        alpha_n_act = op.Add(op.Softplus(self.alpha_n), self.beta)

        # Common sub-expression: beta * x
        beta_x = op.Mul(self.beta, x)

        # Positive branch: softplus(alpha_p) * x² + beta * x
        pos_branch = op.Add(op.Mul(alpha_p_act, op.Mul(x, x)), beta_x)

        # Negative branch: (expm1(clamp(x, max=eps)) - x) * alpha_n + beta * x
        x_clipped = op.Min(x, self.eps)
        neg_branch = op.Add(
            op.Mul(op.Sub(op.Sub(op.Exp(x_clipped), 1.0), x), alpha_n_act),
            beta_x,
        )

        # Select branch based on sign of x
        zero = op.CastLike(0.0, x)
        return op.Where(op.Greater(x, zero), pos_branch, neg_branch)


class ApertusFCMLP(nn.Module):
    """Apertus MLP: up_proj → xIELU → down_proj."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = XIELUActivation()

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.up_proj(op, x)
        x = self.act_fn(op, x)
        return self.down_proj(op, x)


class ApertusCausalLMModel(CausalLMModel):
    """Apertus model with xIELU activation, QK-norm, and custom norm naming."""

    def __init__(self, config: ArchitectureConfig):
        config = dataclasses.replace(config, attn_qk_norm=True)
        super().__init__(config)
        for layer in self.model.layers:
            layer.mlp = ApertusFCMLP(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HF Apertus weights to match ONNX model structure.

        - attention_layernorm → input_layernorm
        - feedforward_layernorm → post_attention_layernorm
        - act_fn scalars (alpha_p, alpha_n, beta, eps): reshape to [1]
        """
        new_state_dict = {}
        for name, tensor in state_dict.items():
            # Rename layer norms
            name = name.replace(".attention_layernorm.", ".input_layernorm.")
            name = name.replace(".feedforward_layernorm.", ".post_attention_layernorm.")

            # xIELU params: HF has scalar [1], ONNX nn.Parameter is [1]
            if name.endswith(
                (
                    ".mlp.act_fn.alpha_p",
                    ".mlp.act_fn.alpha_n",
                    ".mlp.act_fn.beta",
                    ".mlp.act_fn.eps",
                )
            ):
                new_state_dict[name] = tensor.reshape(1)
                continue

            new_state_dict[name] = tensor
        return super().preprocess_weights(new_state_dict)
