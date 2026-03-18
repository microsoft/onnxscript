# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

if TYPE_CHECKING:
    import onnx_ir as ir


class RMSNorm(nn.Module):
    """RMS Layer Normalization using the ONNX RMSNormalization op (opset 23)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self.variance_epsilon = eps

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        return apply_rms_norm(op, hidden_states, self.weight, self.variance_epsilon)


class OffsetRMSNorm(nn.Module):
    """RMSNorm with +1 offset on weight: output = norm(x) * (1.0 + weight).

    Used by Qwen3.5 where the HF checkpoint stores weights initialized
    to zero, and the effective multiplier is (1 + weight).

    Alternative: use standard RMSNorm and add 1.0 in preprocess_weights.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self.variance_epsilon = eps

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        effective_weight = op.Add(self.weight, 1.0)
        return op.RMSNormalization(
            hidden_states,
            effective_weight,
            epsilon=self.variance_epsilon,
            axis=-1,
        )


class GatedRMSNorm(nn.Module):
    """RMSNorm with SiLU gate applied before normalization.

    Computes: weight * RMSNorm(x * SiLU(gate))

    The gate is applied first so it affects the normalization variance,
    matching HuggingFace's MambaRMSNormGated / BambaRMSNormGated.

    Used in Mamba2 and Bamba layers.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self.variance_epsilon = eps

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, gate: ir.Value):
        # Gate first: hidden * SiLU(gate)
        gate_activated = op.Mul(gate, op.Sigmoid(gate))
        gated = op.Mul(hidden_states, gate_activated)
        # Then normalize the gated result
        return op.RMSNormalization(
            gated,
            self.weight,
            epsilon=self.variance_epsilon,
            axis=-1,
        )


class PostGatedRMSNorm(nn.Module):
    """RMSNorm with SiLU gate applied after normalization.

    Computes: weight * RMSNorm(x) * SiLU(gate)

    The gate is applied after normalization so it does NOT affect
    the normalization variance, matching HuggingFace's
    Qwen3NextRMSNormGated.

    Used in Gated DeltaNet layers (Qwen3.5 hybrid models).

    Note: this differs from ``GatedRMSNorm`` which applies the gate
    before normalization (used by Mamba2/Bamba).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self.variance_epsilon = eps

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, gate: ir.Value):
        # Normalize first
        normed = op.RMSNormalization(
            hidden_states,
            self.weight,
            epsilon=self.variance_epsilon,
            axis=-1,
        )
        # Then apply gate: normed * SiLU(gate)
        gate_activated = op.Mul(gate, op.Sigmoid(gate))
        return op.Mul(normed, gate_activated)


def apply_rms_norm(op: builder.OpBuilder, x, weight, eps):
    """Apply RMS normalization using the ONNX RMSNormalization op.

    Args:
        op: The OpBuilder for creating ONNX ops.
        x: Input tensor of shape (batch_size, seq_length, hidden_size).
        weight: Learnable weight tensor of shape (hidden_size,).
        eps: Epsilon value (float or ir.Value scalar tensor).

    Returns:
        Normalized tensor with the same shape as input.
    """
    return op.RMSNormalization(x, weight, epsilon=eps, axis=-1)
