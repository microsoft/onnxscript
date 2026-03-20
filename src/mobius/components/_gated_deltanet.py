# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet: linear attention component for Qwen3.5 hybrid models.

The Gated DeltaNet is a state-space-inspired linear attention mechanism
that alternates with standard full attention layers. It maintains a
fixed-size recurrent state (instead of a growing KV cache) for O(1)
memory per token during decoding.

Architecture per layer:
    1. Linear projections -> Q, K, V, z (gate), b (forget), a (decay)
    2. CausalConv1DWithState — depthwise Conv1D + SiLU + carry state
    3. L2-normalize Q and K
    4. Compute decay: g = -exp(A_log) * softplus(a + dt_bias)
    5. Compute forget: beta = sigmoid(b)
    6. LinearAttention — gated delta-rule recurrence
    7. Gated RMSNorm: output = norm(attn_out) * silu(z)
    8. Output projection

State carried across steps:
    - conv_state: (batch, conv_dim, kernel_size-1) — sliding conv window
    - recurrent_state: (batch, num_v_heads, k_dim, v_dim) — matrix accumulator

The CausalConv1DWithState and LinearAttention ops are defined
as ir.Functions in ``mobius.functions`` and registered on
the model by the task layer.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.components._rms_norm import PostGatedRMSNorm


class _DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution via CausalConv1DWithState function op.

    Wraps a single ``weight`` parameter so that HuggingFace weight names
    (``conv1d.weight``) automatically align with ONNX initializer names.

    The ``forward()`` method calls the ``CausalConv1DWithState``
    function op from the ``pkg.mobius`` domain, passing the
    ``group`` attribute so one function definition works for all channel
    sizes.  Because the module is invoked via ``__call__``,
    ``self.weight`` is automatically realized as a graph initializer
    with the correct qualified name.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter([channels, 1, kernel_size])
        self._channels = channels

    def forward(
        self,
        op: builder.OpBuilder,
        input_val: ir.Value,
        conv_state: ir.Value,
    ):
        """Run CausalConv1DWithState function op.

        Args:
            op: ONNX op builder.
            input_val: (B, D, T) — channels-first input.
            conv_state: (B, D, K-1) — carry state.

        Returns:
            output: (B, D, T) — convolution output with SiLU.
            present_state: (B, D, K-1) — updated carry state.
        """
        # Zero bias — model has no conv bias; the function requires it.
        # CastLike ensures the bias matches the weight dtype (e.g. f16).
        conv_bias = op.Expand(
            op.CastLike(op.Constant(value_float=0.0), self.weight),
            op.Constant(value_ints=[self._channels]),
        )
        return op.CausalConv1DWithState(
            input_val,
            self.weight,
            conv_bias,
            conv_state,
            activation="silu",
            group=self._channels,
            _domain="com.microsoft",
            _outputs=2,
        )


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear attention layer.

    This component implements the single-token decode path for the
    gated delta-rule linear attention.  All complexity (Scan loop,
    GQA head expansion) is inside the ir.Function definitions.
    The forward pass is a clean sequence of op calls.

    The recurrent state replaces the KV cache for these layers.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # QKV projection (fused: Q, K, V in one linear)
        self.in_proj_qkv = Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,
            bias=False,
        )
        # Gating projection (z for silu gating in output norm)
        self.in_proj_z = Linear(self.hidden_size, self.value_dim, bias=False)
        # Beta projection (forget gate)
        self.in_proj_b = Linear(self.hidden_size, self.num_v_heads, bias=False)
        # Alpha projection (decay control)
        self.in_proj_a = Linear(self.hidden_size, self.num_v_heads, bias=False)

        # Causal depthwise Conv1D
        self.conv1d = _DepthwiseConv1d(self.conv_dim, self.conv_kernel_size)

        # Learnable parameters for decay computation
        self.dt_bias = nn.Parameter([self.num_v_heads])
        self.A_log = nn.Parameter([self.num_v_heads])

        # Gated output normalization
        self.norm = PostGatedRMSNorm(self.head_v_dim, eps=config.rms_norm_eps)

        # Output projection
        self.out_proj = Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        recurrent_state: ir.Value,
    ):
        """Forward pass for the Gated DeltaNet layer.

        Clean and loop-free — all complexity (Scan, GQA expansion)
        is inside the ir.Function definitions.

        Args:
            op: ONNX op builder.
            hidden_states: (batch, seq_len, hidden_size).
            conv_state: (batch, conv_dim, kernel_size-1) — carry state.
            recurrent_state: (batch, num_v_heads, k_dim, v_dim) — carry.

        Returns:
            output: (batch, seq_len, hidden_size)
            new_conv_state: (batch, conv_dim, kernel_size-1)
            new_recurrent_state: (batch, num_v_heads, k_dim, v_dim)
        """
        batch_dim = op.Shape(hidden_states, start=0, end=1)

        # === Projections ===
        mixed_qkv = self.in_proj_qkv(op, hidden_states)
        z = self.in_proj_z(op, hidden_states)
        b = self.in_proj_b(op, hidden_states)
        a = self.in_proj_a(op, hidden_states)

        # === CausalConv1DWithState ===
        # Transpose for conv: (batch, conv_dim, seq_len)
        mixed_qkv_t = op.Transpose(mixed_qkv, perm=[0, 2, 1])
        conv_out, new_conv_state = self.conv1d(op, mixed_qkv_t, conv_state)
        # Transpose back: (batch, seq_len, conv_dim)
        conv_out = op.Transpose(conv_out, perm=[0, 2, 1])

        # === Split into Q, K, V ===
        query, key, value = op.Split(
            conv_out,
            op.Constant(value_ints=[self.key_dim, self.key_dim, self.value_dim]),
            axis=-1,
            _outputs=3,
        )

        # === Reshape to head dimensions ===
        seq_dim = op.Shape(hidden_states, start=1, end=2)
        qk_shape = op.Concat(
            batch_dim,
            seq_dim,
            op.Constant(value_ints=[self.num_k_heads]),
            op.Constant(value_ints=[self.head_k_dim]),
            axis=0,
        )
        query = op.Reshape(query, qk_shape)
        key = op.Reshape(key, qk_shape)

        v_shape = op.Concat(
            batch_dim,
            seq_dim,
            op.Constant(value_ints=[self.num_v_heads]),
            op.Constant(value_ints=[self.head_v_dim]),
            axis=0,
        )
        value = op.Reshape(value, v_shape)
        z = op.Reshape(z, v_shape)

        # === L2 normalize Q and K ===
        query = _l2_normalize(op, query)
        key = _l2_normalize(op, key)
        scale = op.CastLike(
            op.Constant(value_float=1.0 / (self.head_k_dim**0.5)),
            query,
        )
        query = op.Mul(query, scale)

        # === Compute gating parameters ===
        # beta: (B, S, num_v_heads)
        beta = op.Sigmoid(b)
        # decay: (B, S, num_v_heads)
        a_plus_dt = op.Add(a, self.dt_bias)
        softplus_val = op.Softplus(a_plus_dt)
        neg_a = op.Neg(op.Exp(op.Cast(self.A_log, to=1)))
        g = op.Mul(neg_a, softplus_val)

        # === LinearAttention ===
        # Transpose from (B, S, H, D) to (B, H, S, D)
        # Q/K keep native num_k_heads; V uses num_v_heads.
        # GQA expansion happens inside the function.
        query_bhsd = op.Transpose(query, perm=[0, 2, 1, 3])
        key_bhsd = op.Transpose(key, perm=[0, 2, 1, 3])
        value_bhsd = op.Transpose(value, perm=[0, 2, 1, 3])

        # beta/decay: (B, S, H) -> (B, H, S)
        beta_bhs = op.Transpose(beta, perm=[0, 2, 1])
        g_bhs = op.Transpose(g, perm=[0, 2, 1])

        output_4d, new_recurrent_state = op.LinearAttention(
            query_bhsd,  # (B, H_kv, S, d_k)
            key_bhsd,  # (B, H_kv, S, d_k)
            value_bhsd,  # (B, H, S, d_v)
            recurrent_state,  # (B, H, d_k, d_v)
            g_bhs,  # (B, H, S) — decay in log-space
            beta_bhs,  # (B, H, S) — update rate
            update_rule="gated_delta",
            _domain="com.microsoft",
            _outputs=2,
        )
        # output_4d: (B, H, S, d_v) -> (B, S, H, d_v)
        output_per_head = op.Transpose(output_4d, perm=[0, 2, 1, 3])

        # === Gated RMSNorm ===
        flat_shape = op.Constant(value_ints=[-1, self.head_v_dim])
        output_flat = op.Reshape(output_per_head, flat_shape)
        z_flat = op.Reshape(z, flat_shape)
        normed = self.norm(op, output_flat, z_flat)

        # === Output projection ===
        out_3d_shape = op.Concat(
            batch_dim,
            seq_dim,
            op.Constant(value_ints=[self.value_dim]),
            axis=0,
        )
        output_3d = op.Reshape(normed, out_3d_shape)
        output = self.out_proj(op, output_3d)

        return output, new_conv_state, new_recurrent_state


def _l2_normalize(op: builder.OpBuilder, x, eps: float = 1e-6):
    """L2 normalize along the last dimension."""
    sq = op.Mul(x, x)
    sq_sum = op.ReduceSum(sq, [-1], keepdims=True)
    eps_val = op.CastLike(op.Constant(value_float=eps), x)
    inv_norm = op.Reciprocal(op.Sqrt(op.Add(sq_sum, eps_val)))
    return op.Mul(x, inv_norm)
