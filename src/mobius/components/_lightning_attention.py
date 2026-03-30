# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Lightning Attention component for MiniMax hybrid models.

MiniMax Lightning Attention uses a pure retention (gated) linear attention
recurrence without delta correction:

    state_t = exp(-slope_rate) · state_{t-1} + k_t ⊗ v_t
    output_t = q_t @ state_t

The per-head decay rate is derived analytically from layer_idx and
num_hidden_layers — it is a fixed scalar (not learned per-token). This
maps to the ONNX LinearAttention ``update_rule="gated"`` with a constant
decay tensor.

Key differences from GatedDeltaNet:
- No CausalConvWithState preprocessing
- No L2 normalization of Q/K
- Fused QKV with SiLU activation applied before split
- sigmoid(output_gate) gating (not silu(z))
- Static decay from slope_rate buffer
- Square state matrix: (B, H, d_k, d_k) since d_v = d_k = head_dim

HuggingFace reference: MiniMaxLightningAttention in
  transformers.models.minimax.modeling_minimax
"""

from __future__ import annotations

import math

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.components._rms_norm import RMSNorm

DOMAIN = "com.microsoft"


class LightningAttention(nn.Module):
    """MiniMax Lightning Attention: pure retention linear attention layer.

    Recurrence (token-level):
        S_t = exp(-slope_rate) * S_{t-1} + k_t ⊗ v_t
        o_t = q_t @ S_t

    The decay is a per-head static scalar computed from layer_idx and
    num_hidden_layers. It is passed to the LinearAttention "gated" function
    as a constant broadcast tensor of shape (B, T, num_heads).

    State: (B, num_heads, head_dim, head_dim) — square matrix accumulator.

    Args:
        config: Architecture config. Uses hidden_size, num_attention_heads,
            rms_norm_eps, num_hidden_layers.
        layer_idx: Zero-based layer index, used to compute slope_rate.
    """

    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self._dtype = config.dtype

        # Fused QKV projection — SiLU applied to all 3*H*d_k before split
        self.qkv_proj = Linear(
            config.hidden_size,
            self.num_heads * self.head_dim * 3,
            bias=False,
        )
        self.out_proj = Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        # output_gate: sigmoid gate applied to the normalized attention output
        self.output_gate = Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.norm = RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)

        # Per-head log-space decay values (negative, so exp < 1)
        # HF: slope_rate[h] = base^(h+1) * factor
        #     base = 1 / (2^(8/H)), factor = 1 - layer_idx/(L-1+ε) + ε
        self._decay_log: list[float] = _compute_decay_log(
            layer_idx, config.num_hidden_layers, self.num_heads
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        recurrent_state: ir.Value,
    ):
        """Lightning Attention forward.

        Args:
            op: ONNX op builder.
            hidden_states: (B, T, hidden_size).
            recurrent_state: (B, num_heads, head_dim, head_dim) — carry state.

        Returns:
            output: (B, T, hidden_size).
            new_recurrent_state: (B, num_heads, head_dim, head_dim).
        """
        batch_dim = op.Shape(hidden_states, start=0, end=1)  # (1,) int64
        seq_dim = op.Shape(hidden_states, start=1, end=2)  # (1,) int64

        # Fused QKV: project then apply SiLU before split (matches HF)
        # qkv: (B, T, 3 * num_heads * head_dim)
        qkv = self.qkv_proj(op, hidden_states)
        qkv = op.Mul(qkv, op.Sigmoid(qkv))  # SiLU: x * sigmoid(x)

        # Split into Q, K, V: each (B, T, num_heads * head_dim)
        head_total = self.num_heads * self.head_dim
        query, key, value = op.Split(
            qkv,
            op.Constant(value_ints=[head_total, head_total, head_total]),
            axis=-1,
            _outputs=3,
        )

        # Static decay tensor: (B, T, num_heads) with constant per-head values.
        # Each decay[h] = -slope_rate[h] in log-space → exp(decay[h]) < 1.
        # The LinearAttention "gated" rule applies exp(g_t) to the state at each step.
        decay_per_head = op.CastLike(
            op.Constant(value_floats=self._decay_log),  # (num_heads,)
            hidden_states,
        )
        # Reshape to (1, 1, num_heads) then broadcast to (B, T, num_heads)
        decay_1 = op.Reshape(
            decay_per_head,
            op.Constant(value_ints=[1, 1, self.num_heads]),
        )
        expand_to = op.Concat(
            batch_dim, seq_dim, op.Constant(value_ints=[self.num_heads]), axis=0
        )
        decay = op.Expand(decay_1, expand_to)  # (B, T, num_heads)

        # LinearAttention "gated": S_t = exp(g_t) * S_{t-1} + k_t ⊗ v_t
        # scale = 1/sqrt(head_dim) for proper scaling
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out, new_state = op.LinearAttention(
            query,
            key,
            value,
            recurrent_state,
            decay,
            update_rule="gated",
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=scale,
            _domain=DOMAIN,
            _outputs=2,
        )
        # attn_out: (B, T, num_heads * head_dim)

        # RMSNorm + sigmoid output gate (matches HF output pipeline)
        attn_out = self.norm(op, attn_out)
        gate = op.Sigmoid(self.output_gate(op, hidden_states))
        attn_out = op.Mul(gate, attn_out)

        output = self.out_proj(op, attn_out)
        return output, new_state


def _compute_decay_log(layer_idx: int, num_layers: int, num_heads: int) -> list[float]:
    """Compute per-head log-space decay values for Lightning Attention.

    Returns negative values so that exp(decay[h]) = exp(-slope_rate[h]) < 1.
    Matches HF ``MiniMaxLightningAttention.get_slope_rate()``.
    """
    base = 1.0 / (2.0 ** (8.0 / num_heads))
    factor = 1.0 - layer_idx / (num_layers - 1.0 + 1e-5) + 1e-5
    return [-(base ** (h + 1)) * factor for h in range(num_heads)]
