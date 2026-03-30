# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Mamba block component: Conv1D → SSM → gated output projection.

Implements the standard Mamba layer used in Mamba, Jamba, Bamba,
FalconMamba, and related architectures. Composes a causal depthwise
Conv1D with the SelectiveScan SSM and a SiLU-gated output path.

Architecture per layer:
    1. in_proj: x → (x_branch, z_gate)  [expansion to d_inner]
    2. Causal depthwise Conv1D on x_branch
    3. SiLU activation
    4. Selective scan (SSM) with recurrent state
    5. Output gating: y * SiLU(z)
    6. out_proj: project back to d_model

State carried across steps:
    conv_state:  (batch, d_inner, conv_kernel - 1)
    ssm_state:   (batch, d_inner, d_state)

HuggingFace reference: ``MambaMixer``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import INT64_MAX, Linear
from mobius.components._rms_norm import GatedRMSNorm
from mobius.components._ssm import Mamba2Scan, SelectiveScan

if TYPE_CHECKING:
    import onnx_ir as ir


class _DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution with optional bias.

    Each input channel is convolved with its own kernel (groups=channels).
    Used for causal convolution in the Mamba block.
    """

    def __init__(self, channels: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter([channels, 1, kernel_size])
        self.bias = nn.Parameter([channels]) if bias else None
        self._kernel_size = kernel_size
        self._channels = channels

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: (batch, channels, seq_len)
        result = op.Conv(
            x,
            self.weight,
            kernel_shape=[self._kernel_size],
            strides=[1],
            pads=[0, 0],
            group=self._channels,
        )
        if self.bias is not None:
            # bias: (channels,) → (1, channels, 1) for broadcasting
            bias_3d = op.Unsqueeze(self.bias, [0, 2])
            result = op.Add(result, bias_3d)
        return result


class MambaBlock(nn.Module):
    """Standard Mamba layer: input projection → Conv1D → SSM → gated output.

    Args:
        d_model: Model hidden dimension.
        d_inner: Expanded inner dimension (typically ``expand * d_model``).
        d_state: SSM state dimension (typically 16).
        dt_rank: Rank of the SSM time-step projection.
            Defaults to ``ceil(d_model / 16)`` (Mamba convention).
        conv_kernel: Causal Conv1D kernel size (typically 4).
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int | None = None,
        conv_kernel: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.conv_kernel = conv_kernel
        # Default dt_rank: ceil(d_model / 16) per Mamba convention
        self.dt_rank = dt_rank if dt_rank is not None else -(-d_model // 16)

        # Input projection: d_model → 2*d_inner (x_branch + z_gate)
        self.in_proj = Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise Conv1D (with bias, matching HuggingFace)
        self.conv1d = _DepthwiseConv1d(d_inner, conv_kernel, bias=True)

        # Core SSM component
        self.ssm = SelectiveScan(d_inner, d_state, self.dt_rank)

        # Output projection: d_inner → d_model
        self.out_proj = Linear(d_inner, d_model, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Single-token forward pass for the Mamba layer.

        Args:
            op: ONNX op builder.
            hidden_states: (batch, 1, d_model) — single token input.
            conv_state: (batch, d_inner, conv_kernel-1) — carry state.
            ssm_state: (batch, d_inner, d_state) — carry state.

        Returns:
            output: (batch, 1, d_model)
            new_conv_state: (batch, d_inner, conv_kernel-1)
            new_ssm_state: (batch, d_inner, d_state)
        """
        # --- Step 1: Input projection ---
        # projected: (batch, 1, 2*d_inner)
        projected = self.in_proj(op, hidden_states)

        # Split into x_branch and z_gate along last dim
        x_branch, z_gate = op.Split(
            projected,
            op.Constant(value_ints=[self.d_inner, self.d_inner]),
            axis=-1,
            _outputs=2,
        )
        # x_branch: (batch, 1, d_inner)
        # z_gate:   (batch, 1, d_inner)

        # --- Step 2: Causal Conv1D with state update ---
        # Transpose for conv: (batch, d_inner, 1)
        x_t = op.Transpose(x_branch, perm=[0, 2, 1])

        # Concatenate conv state + new token: (batch, d_inner, conv_kernel)
        conv_input = op.Concat(conv_state, x_t, axis=2)

        # Update conv state: drop oldest, keep last (conv_kernel-1)
        new_conv_state = op.Slice(
            conv_input,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[INT64_MAX]),
            op.Constant(value_ints=[2]),
        )

        # Apply depthwise conv: (batch, d_inner, 1)
        conv_out = self.conv1d(op, conv_input)

        # --- Step 3: SiLU activation ---
        conv_out = op.Mul(conv_out, op.Sigmoid(conv_out))

        # Transpose back: (batch, 1, d_inner)
        x_ssm = op.Transpose(conv_out, perm=[0, 2, 1])

        # --- Step 4: Selective scan ---
        y, new_ssm_state = self.ssm(op, x_ssm, ssm_state)
        # y: (batch, 1, d_inner)

        # --- Step 5: Output gating: y * SiLU(z) ---
        z_activated = op.Mul(z_gate, op.Sigmoid(z_gate))
        gated = op.Mul(y, z_activated)

        # --- Step 6: Output projection ---
        # output: (batch, 1, d_model)
        output = self.out_proj(op, gated)

        return output, new_conv_state, new_ssm_state


class Mamba2Block(nn.Module):
    """Mamba2/SSD block: in_proj -> Conv1D -> multi-head SSM -> gated norm.

    Key differences from MambaBlock (Mamba1):
    - in_proj outputs [gate, xBC, dt] instead of [x, z]
    - Conv1D on wider xBC (conv_dim channels)
    - Multi-head SSM with grouped B/C
    - GatedRMSNorm instead of SiLU gating
    - dt direct from in_proj (no rank reduction), just bias

    HuggingFace reference: ``BambaMixer``.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        num_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int = 1,
        conv_kernel: int = 4,
        conv_bias: bool = True,
        proj_bias: bool = False,
        eps: float = 1e-5,
        norm_group_size: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel

        self.conv_dim = d_inner + 2 * n_groups * d_state

        proj_size = d_inner + self.conv_dim + num_heads
        self.in_proj = Linear(d_model, proj_size, bias=proj_bias)
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel, bias=conv_bias)
        self.ssm = Mamba2Scan(num_heads, d_head, d_state, n_groups)
        self.norm = GatedRMSNorm(d_inner, eps=eps, group_size=norm_group_size)
        self.out_proj = Linear(d_inner, d_model, bias=proj_bias)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Single-token forward pass for the Mamba2 block.

        Args:
            op: ONNX op builder.
            hidden_states: (batch, 1, d_model)
            conv_state: (batch, conv_dim, conv_kernel-1)
            ssm_state: (batch, num_heads, d_head, d_state)

        Returns:
            output: (batch, 1, d_model)
            new_conv_state: (batch, conv_dim, conv_kernel-1)
            new_ssm_state: (batch, num_heads, d_head, d_state)
        """
        # Step 1: Input projection -> gate, xBC, dt
        projected = self.in_proj(op, hidden_states)
        gate, x_bc, dt = op.Split(
            projected,
            op.Constant(value_ints=[self.d_inner, self.conv_dim, self.num_heads]),
            axis=-1,
            _outputs=3,
        )

        # Step 2: Causal Conv1D with state update
        x_bc_t = op.Transpose(x_bc, perm=[0, 2, 1])
        conv_input = op.Concat(conv_state, x_bc_t, axis=2)
        new_conv_state = op.Slice(
            conv_input,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[INT64_MAX]),
            op.Constant(value_ints=[2]),
        )
        conv_out = self.conv1d(op, conv_input)

        # Step 3: SiLU activation
        conv_out = op.Mul(conv_out, op.Sigmoid(conv_out))
        x_bc_activated = op.Transpose(conv_out, perm=[0, 2, 1])

        # Step 4: Split xBC -> hidden, B, C
        groups_state = self.n_groups * self.d_state
        hidden_x, b_mat, c_mat = op.Split(
            x_bc_activated,
            op.Constant(value_ints=[self.d_inner, groups_state, groups_state]),
            axis=-1,
            _outputs=3,
        )

        # Squeeze seq dim for SSM
        hidden_flat = op.Squeeze(hidden_x, [1])
        dt_flat = op.Squeeze(dt, [1])
        b_flat = op.Squeeze(b_mat, [1])
        c_flat = op.Squeeze(c_mat, [1])

        # Step 5: Multi-head selective scan
        y, new_ssm_state = self.ssm(op, hidden_flat, dt_flat, b_flat, c_flat, ssm_state)

        # Step 6: Gated RMSNorm
        gate_flat = op.Squeeze(gate, [1])
        y_normed = self.norm(op, y, gate_flat)

        # Restore seq dim
        y_3d = op.Unsqueeze(y_normed, [1])

        # Step 7: Output projection
        output = self.out_proj(op, y_3d)

        return output, new_conv_state, new_ssm_state
