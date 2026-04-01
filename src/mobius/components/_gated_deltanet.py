# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet: linear attention component for Qwen3.5 hybrid models.

The Gated DeltaNet is a state-space-inspired linear attention mechanism
that alternates with standard full attention layers. It maintains a
fixed-size recurrent state (instead of a growing KV cache) for O(1)
memory per token during decoding.

Architecture per layer:
    1. Linear projections -> Q, K, V, z (gate), b (forget), a (decay)
    2. CausalConvWithState — depthwise Conv1D + SiLU + carry state
    3. L2-normalize Q and K (Sqrt/ReduceSumSquare/Div decomposition)
    4. Compute decay: g = -exp(A_log) * softplus(a + dt_bias)
    5. Compute forget: beta = sigmoid(b)
    6. LinearAttention — gated delta-rule recurrence
    7. Gated RMSNorm: output = norm(attn_out) * silu(z)
    8. Output projection

State carried across steps:
    - conv_state: (batch, conv_dim, kernel_size-1) — sliding conv window
    - recurrent_state: (batch, num_v_heads, k_dim, v_dim) — matrix accumulator

The CausalConvWithState and LinearAttention ops are defined
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
    """Depthwise 1D convolution via CausalConvWithState function op.

    Wraps a single ``weight`` parameter so that HuggingFace weight names
    (``conv1d.weight``) automatically align with ONNX initializer names.

    The ``forward()`` method calls the ``CausalConvWithState``
    function op in the ``com.microsoft`` domain (registered as an
    ``ir.Function`` by the task layer). The function is specialized
    for depthwise convolution with ``group = channels`` baked in at
    construction time from the ``channels`` argument, so the caller
    does not supply ``group`` and it is not derived from the runtime
    input shape. Because the module is invoked via ``__call__``,
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
        """Run CausalConvWithState function op.

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
        return op.CausalConvWithState(
            input_val,
            self.weight,
            conv_bias,
            conv_state,
            activation="silu",
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
        self._dtype = config.dtype
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

        # === CausalConvWithState ===
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
        # query, key: [B, T, key_dim] (3D)
        # value: [B, T, value_dim] (3D)

        # === L2 normalize Q and K (per-head, along d_k) ===
        # Reshape to 4D for per-head normalization, then back to 3D.
        seq_dim = op.Shape(hidden_states, start=1, end=2)
        qk_4d_shape = op.Concat(
            batch_dim,
            seq_dim,
            op.Constant(value_ints=[self.num_k_heads, self.head_k_dim]),
            axis=0,
        )
        qk_3d_shape = op.Concat(
            batch_dim,
            seq_dim,
            op.Constant(value_ints=[self.key_dim]),
            axis=0,
        )
        # L2-normalize query and key per head along head_k_dim (axis=-1).
        # Decomposed form of op.LpNormalization(x, axis=-1, p=2).
        # TODO: Use op.LpNormalization directly once ORT >=1.25 supports it.
        q_4d = op.Reshape(query, qk_4d_shape)  # (B, T, num_k_heads, head_k_dim)
        q_l2 = op.Sqrt(
            op.ReduceSumSquare(q_4d, [-1], keepdims=1)
        )  # (B, T, num_k_heads, 1) — L2 norm per head
        query = op.Reshape(op.Div(q_4d, q_l2), qk_3d_shape)  # (B, T, key_dim)

        k_4d = op.Reshape(key, qk_4d_shape)  # (B, T, num_k_heads, head_k_dim)
        k_l2 = op.Sqrt(
            op.ReduceSumSquare(k_4d, [-1], keepdims=1)
        )  # (B, T, num_k_heads, 1) — L2 norm per head
        key = op.Reshape(op.Div(k_4d, k_l2), qk_3d_shape)  # (B, T, key_dim)

        # === Compute gating parameters ===
        # beta: (B, T, num_v_heads)
        beta = op.Sigmoid(b)
        # Compute decay: g = -exp(A_log) * softplus(a + dt_bias).
        # fp16 Exp overflows at ~11.09; bf16 has the same exponent range as fp32
        # (8-bit exponent) so no upcast is needed for bf16 or fp32.
        # Mirrors HF: -A_log.float().exp() * softplus(a.float() + dt_bias).
        if self._dtype == ir.DataType.FLOAT16:
            # Upcast to fp32 to avoid fp16 Exp/Softplus overflow.
            a_f32 = op.Cast(a, to=ir.DataType.FLOAT)
            dt_bias_f32 = op.Cast(self.dt_bias, to=ir.DataType.FLOAT)
            a_log_f32 = op.Cast(self.A_log, to=ir.DataType.FLOAT)
            softplus_val = op.Softplus(op.Add(a_f32, dt_bias_f32))
            neg_a = op.Neg(op.Exp(a_log_f32))
            g = op.CastLike(op.Mul(neg_a, softplus_val), a)  # cast back to fp16
        else:
            # bf16/fp32: sufficient exponent range, compute natively.
            softplus_val = op.Softplus(op.Add(a, self.dt_bias))
            neg_a = op.Neg(op.Exp(self.A_log))
            g = op.Mul(neg_a, softplus_val)
        # g: (B, T, num_v_heads)

        # === LinearAttention ===
        # beta: (B, T, num_v_heads) — already 3D, matches (B, T, kv_num_heads)
        # decay g: (B, T, num_v_heads) — per-head scalar decay (d_k=1),
        #   matches (B, T, kv_num_heads * 1) = (B, T, kv_num_heads)

        output_3d, new_recurrent_state = op.LinearAttention(
            query,  # (B, T, num_k_heads * head_k_dim)
            key,  # (B, T, num_k_heads * head_k_dim)
            value,  # (B, T, num_v_heads * head_v_dim)
            recurrent_state,  # (B, num_v_heads, d_k, d_v)
            g,  # (B, T, num_v_heads) — decay in log-space, broadcasts over d_k
            beta,  # (B, T, num_v_heads) — update rate
            scale=1.0 / (self.head_k_dim**0.5),
            q_num_heads=self.num_k_heads,
            kv_num_heads=self.num_v_heads,
            _domain="com.microsoft",
            _outputs=2,
        )
        # output_3d: (B, T, num_v_heads * d_v) — already 3D

        # === Gated RMSNorm ===
        flat_shape = op.Constant(value_ints=[-1, self.head_v_dim])
        output_flat = op.Reshape(output_3d, flat_shape)
        z_flat = op.Reshape(z, flat_shape)
        normed = self.norm(op, output_flat, z_flat)

        # === Output projection ===
        out_3d_shape = op.Concat(
            batch_dim,
            seq_dim,
            op.Constant(value_ints=[self.value_dim]),
            axis=0,
        )
        output = op.Reshape(normed, out_3d_shape)
        output = self.out_proj(op, output)

        return output, new_conv_state, new_recurrent_state
