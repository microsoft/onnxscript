# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._flags import flags


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

    When ``group_size`` is set, variance is computed per group
    (matching HF's ``Zamba2RMSNormGated``). This is used by
    NemotronH where group_size = d_inner // n_groups.

    Used in Mamba2, Bamba, and NemotronH layers.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        group_size: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter([hidden_size])
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value, gate: ir.Value):
        # SiLU gating in fp32 for precision, matching HF.
        h_f32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        g_f32 = op.Cast(gate, to=ir.DataType.FLOAT)
        gate_activated = op.Mul(g_f32, op.Sigmoid(g_f32))
        gated = op.Mul(h_f32, gate_activated)

        if self.group_size is not None and self.group_size < self.hidden_size:
            # Grouped RMSNorm: reshape to (batch, n_groups, group_size),
            # normalize within each group, then reshape back.
            n_groups = self.hidden_size // self.group_size
            if flags.ort_cuda_grouped_rmsnorm_workaround:
                # ORT ≤1.24.4 CUDA kernel for RMSNormalization produces
                # wrong results when scale is 2D. Decompose into basic
                # ops as a workaround.
                grouped = op.Reshape(
                    gated,
                    op.Constant(value_ints=[0, n_groups, self.group_size]),
                )
                variance = op.ReduceMean(
                    op.Mul(grouped, grouped),
                    axes=[-1],
                    keepdims=True,
                )
                rnorm = op.Reciprocal(
                    op.Sqrt(op.Add(variance, self.variance_epsilon)),
                )
                normed = op.Mul(grouped, rnorm)
                normed = op.Reshape(
                    normed,
                    op.Constant(value_ints=[0, self.hidden_size]),
                )
                normed = op.Mul(
                    normed,
                    op.Cast(self.weight, to=ir.DataType.FLOAT),
                )
            else:
                # Cast gated back to native dtype; RMSNormalization's
                # stash_type=1 handles internal fp32 for variance.
                gated = op.CastLike(gated, hidden_states)
                grouped = op.Reshape(
                    gated,
                    op.Constant(value_ints=[0, n_groups, self.group_size]),
                )
                weight_grouped = op.Reshape(
                    self.weight,
                    op.Constant(value_ints=[n_groups, self.group_size]),
                )
                normed = op.RMSNormalization(
                    grouped,
                    weight_grouped,
                    epsilon=self.variance_epsilon,
                    axis=-1,
                )
                normed = op.Reshape(
                    normed,
                    op.Constant(value_ints=[0, self.hidden_size]),
                )
        else:
            # Standard RMSNorm over the full dimension.
            # Cast gated back to native dtype; stash_type=1 handles fp32.
            gated = op.CastLike(gated, hidden_states)
            normed = op.RMSNormalization(
                gated,
                self.weight,
                epsilon=self.variance_epsilon,
                axis=-1,
            )
        return op.CastLike(normed, hidden_states)


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
        # RMSNorm uses stash_type=1 internally for fp32 variance.
        normed = op.RMSNormalization(
            hidden_states,
            self.weight,
            epsilon=self.variance_epsilon,
            stash_type=1,
            axis=-1,
        )
        # Apply gate in fp32: normed * SiLU(gate), then cast back.
        # Matches HF Qwen3_5RMSNormGated which does gate.to(float32).
        g_f32 = op.Cast(gate, to=ir.DataType.FLOAT)
        gate_activated = op.Mul(g_f32, op.Sigmoid(g_f32))
        result = op.Mul(op.Cast(normed, to=ir.DataType.FLOAT), gate_activated)
        return op.CastLike(result, hidden_states)


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
