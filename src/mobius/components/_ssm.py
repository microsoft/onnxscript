# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Selective State Space Model (S6) component for Mamba architectures.

Implements the core selective scan recurrence from the Mamba paper.
Data-dependent B, C matrices enable content-aware state transitions,
distinguishing this from classical (fixed-parameter) SSMs.

Single-token decode recurrence:
    dt = softplus(dt_proj(x_proj(x)[:dt_rank]))
    B  = x_proj(x)[dt_rank : dt_rank + d_state]
    C  = x_proj(x)[dt_rank + d_state :]
    dA = exp(dt · A)                      # discretised decay
    state = dA * state + (dt · B) ⊗ x     # state update
    y = C · state + D * x                 # readout + skip

State carried across steps:
    ssm_state: (batch, d_inner, d_state)

HuggingFace reference: ``MambaMixer`` (SSM portion).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class SelectiveScan(nn.Module):
    """Core selective scan (S6) operation.

    Args:
        d_inner: Expanded hidden dimension (``expand * d_model``).
        d_state: SSM state dimension (typically 16).
        dt_rank: Rank of the time-step projection.
    """

    def __init__(self, d_inner: int, d_state: int, dt_rank: int):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Project x → (dt_raw, B, C): dt_rank + 2*d_state outputs
        self.x_proj = Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        # Project dt from reduced rank back to d_inner
        self.dt_proj = Linear(dt_rank, d_inner, bias=True)

        # A_log: (d_inner, d_state) — log of state decay matrix
        self.A_log = nn.Parameter([d_inner, d_state])
        # D: (d_inner,) — skip connection / feedthrough
        self.D = nn.Parameter([d_inner])

    def _project_ssm_params(self, op: builder.OpBuilder, x_db):
        """Split x_proj output into dt_raw, B, C.

        Subclasses can override to apply extra transformations (e.g.
        layernorms on B/C/dt as in Jamba).

        Args:
            op: ONNX op builder.
            x_db: (batch, 1, dt_rank + 2*d_state) — x_proj output.

        Returns:
            Tuple of (dt_raw, b_mat, c_mat).
        """
        dt_raw, b_mat, c_mat = op.Split(
            x_db,
            op.Constant(value_ints=[self.dt_rank, self.d_state, self.d_state]),
            axis=-1,
            _outputs=3,
        )
        return dt_raw, b_mat, c_mat

    def forward(
        self,
        op: builder.OpBuilder,
        x: ir.Value,
        ssm_state: ir.Value,
    ):
        """Single-token selective scan step.

        Args:
            op: ONNX op builder.
            x: (batch, 1, d_inner) — input after conv1d + activation.
            ssm_state: (batch, d_inner, d_state) — carry state.

        Returns:
            y: (batch, 1, d_inner) — output.
            new_ssm_state: (batch, d_inner, d_state) — updated state.
        """
        # --- Project x to get dt, B, C ---
        # x_db: (batch, 1, dt_rank + 2*d_state)
        x_db = self.x_proj(op, x)

        # Split into dt, B, C (hookable for subclass overrides)
        dt_raw, b_mat, c_mat = self._project_ssm_params(op, x_db)
        # dt_raw: (batch, 1, dt_rank)
        # b_mat: (batch, 1, d_state)
        # c_mat: (batch, 1, d_state)

        # --- Compute time step dt ---
        # dt: (batch, 1, d_inner) via rank projection + softplus
        dt = self.dt_proj(op, dt_raw)
        dt = op.Softplus(dt)

        # --- Discretize state decay matrix A ---
        # a_neg = -exp(A_log): (d_inner, d_state)
        a_neg = op.Neg(op.Exp(op.Cast(self.A_log, to=1)))

        # Broadcast: dt (batch,1,d_inner,1) * A (1,1,d_inner,d_state)
        dt_4d = op.Unsqueeze(dt, [-1])  # (batch, 1, d_inner, 1)
        a_4d = op.Unsqueeze(a_neg, [0, 1])  # (1, 1, d_inner, d_state)
        # da = exp(dt * A): (batch, 1, d_inner, d_state)
        da = op.Exp(op.Mul(dt_4d, a_4d))

        # --- Input contribution: dt * B * x ---
        b_4d = op.Unsqueeze(b_mat, [2])  # (batch, 1, 1, d_state)
        dt_b = op.Mul(dt_4d, b_4d)  # (batch, 1, d_inner, d_state)
        x_4d = op.Unsqueeze(x, [-1])  # (batch, 1, d_inner, 1)
        db_x = op.Mul(dt_b, x_4d)  # (batch, 1, d_inner, d_state)

        # --- Squeeze seq dim (single token) ---
        da_t = op.Squeeze(da, [1])  # (batch, d_inner, d_state)
        db_x_t = op.Squeeze(db_x, [1])  # (batch, d_inner, d_state)

        # --- State update: h = dA * h_prev + dBx ---
        new_ssm_state = op.Add(op.Mul(da_t, ssm_state), db_x_t)

        # --- Readout: y = C · h ---
        c_t = op.Squeeze(c_mat, [1])  # (batch, d_state)
        c_3d = op.Unsqueeze(c_t, [1])  # (batch, 1, d_state)
        # h: (batch, d_inner, d_state), C: (batch, 1, d_state)
        # → element-wise mul then sum over d_state → (batch, d_inner)
        y = op.ReduceSum(op.Mul(new_ssm_state, c_3d), [-1], keepdims=False)

        # --- Skip connection: y += D * x ---
        x_t = op.Squeeze(x, [1])  # (batch, d_inner)
        y = op.Add(y, op.Mul(op.Cast(self.D, to=1), x_t))

        # Restore seq dim: (batch, 1, d_inner)
        y = op.Unsqueeze(y, [1])

        return y, new_ssm_state


class JambaSelectiveScan(SelectiveScan):
    """Jamba variant of SelectiveScan with layernorms on dt, B, C.

    Jamba applies additional RMSNorm to each of the SSM parameters
    (dt, B, C) after splitting from x_proj, before the dt projection.

    HuggingFace reference: ``JambaMambaMixer`` (SSM path).
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int,
        dt_rank: int,
        layer_norm_epsilon: float = 1e-5,
    ):
        super().__init__(d_inner, d_state, dt_rank)
        self.dt_layernorm = _RMSNorm(dt_rank, eps=layer_norm_epsilon)
        self.b_layernorm = _RMSNorm(d_state, eps=layer_norm_epsilon)
        self.c_layernorm = _RMSNorm(d_state, eps=layer_norm_epsilon)

    def _project_ssm_params(self, op, x_db):
        """Split + layernorm on dt, B, C."""
        dt_raw, b_mat, c_mat = op.Split(
            x_db,
            op.Constant(value_ints=[self.dt_rank, self.d_state, self.d_state]),
            axis=-1,
            _outputs=3,
        )
        dt_raw = self.dt_layernorm(op, dt_raw)
        b_mat = self.b_layernorm(op, b_mat)
        c_mat = self.c_layernorm(op, c_mat)
        return dt_raw, b_mat, c_mat


class Mamba2Scan(nn.Module):
    """Multi-head selective scan for Mamba2/SSD architecture.

    Mamba2 uses a multi-head structure where each head independently
    scans its own ``d_head`` dimensions. B and C are shared across
    groups of heads (``n_groups``).

    Single-token decode recurrence per head ``h``:
        dt = softplus(dt_input + dt_bias)
        dA = exp(dt * A)
        state[h] = dA * state[h] + (dt * B) * x[h]
        y[h] = C . state[h] + D * x[h]

    State: (batch, num_heads, d_head, d_state)

    HuggingFace reference: ``BambaMixer`` (SSM portion).
    """

    def __init__(
        self,
        num_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.heads_per_group = num_heads // n_groups

        self.A_log = nn.Parameter([num_heads])
        self.D = nn.Parameter([num_heads])
        self.dt_bias = nn.Parameter([num_heads])

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        dt_input: ir.Value,
        b_mat: ir.Value,
        c_mat: ir.Value,
        ssm_state: ir.Value,
    ):
        """Single-token Mamba2 multi-head scan step.

        Args:
            op: ONNX op builder.
            hidden_states: (batch, num_heads * d_head) -- from conv output.
            dt_input: (batch, num_heads) -- time step from in_proj.
            b_mat: (batch, n_groups * d_state) -- B from conv output.
            c_mat: (batch, n_groups * d_state) -- C from conv output.
            ssm_state: (batch, num_heads, d_head, d_state) -- carry.

        Returns:
            y: (batch, num_heads * d_head) -- output.
            new_ssm_state: (batch, num_heads, d_head, d_state).
        """
        # dt = softplus(dt_input + dt_bias): (batch, num_heads)
        dt = op.Add(dt_input, op.Cast(self.dt_bias, to=1))
        dt = op.Softplus(dt)

        # A = -exp(A_log): (num_heads,)
        a_neg = op.Neg(op.Exp(op.Cast(self.A_log, to=1)))

        # Broadcast for state update
        dt_4d = op.Unsqueeze(dt, [2, 3])  # (batch, num_heads, 1, 1)
        a_2d = op.Unsqueeze(a_neg, [0])  # (1, num_heads)
        a_4d = op.Unsqueeze(a_2d, [2, 3])  # (1, num_heads, 1, 1)
        da = op.Exp(op.Mul(dt_4d, a_4d))

        # Reshape hidden: (batch, num_heads, d_head)
        hidden_shape = op.Constant(value_ints=[0, self.num_heads, self.d_head])
        hidden_3d = op.Reshape(hidden_states, hidden_shape)

        # Expand B from groups to heads
        b_shape = op.Constant(value_ints=[0, self.n_groups, 1, self.d_state])
        b_4d = op.Reshape(b_mat, b_shape)
        b_expand_shape = op.Constant(value_ints=[1, 1, self.heads_per_group, 1])
        b_expanded = op.Expand(b_4d, b_expand_shape)
        b_heads_shape = op.Constant(value_ints=[0, self.num_heads, self.d_state])
        b_heads = op.Reshape(b_expanded, b_heads_shape)
        b_ssm = op.Unsqueeze(b_heads, [2])

        # dBx: dt * B * x
        dt_b = op.Mul(dt_4d, b_ssm)  # (batch, num_heads, 1, d_state)
        x_4d = op.Unsqueeze(hidden_3d, [3])
        db_x = op.Mul(dt_b, x_4d)  # (batch, num_heads, d_head, d_state)

        # State update: h = dA * h_prev + dBx
        new_ssm_state = op.Add(op.Mul(da, ssm_state), db_x)

        # Readout: y = C . h + D * x
        c_4d = op.Reshape(c_mat, b_shape)
        c_expanded = op.Expand(c_4d, b_expand_shape)
        c_heads = op.Reshape(c_expanded, b_heads_shape)
        c_ssm = op.Unsqueeze(c_heads, [2])

        y = op.ReduceSum(
            op.Mul(new_ssm_state, c_ssm), [-1], keepdims=False
        )  # (batch, num_heads, d_head)

        # Skip: y += D * x
        d_3d = op.Unsqueeze(op.Cast(self.D, to=1), [0, 2])
        y = op.Add(y, op.Mul(d_3d, hidden_3d))

        # Flatten: (batch, num_heads * d_head)
        flat_shape = op.Constant(value_ints=[0, self.num_heads * self.d_head])
        y = op.Reshape(y, flat_shape)

        return y, new_ssm_state


class _RMSNorm(nn.Module):
    """Lightweight RMSNorm for SSM params (avoids circular import)."""

    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter([size])
        self._eps = eps

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        variance = op.ReduceMean(op.Mul(x, x), [-1], keepdims=True)
        x_normed = op.Div(
            x,
            op.Sqrt(op.Add(variance, op.Constant(value_float=self._eps))),
        )
        return op.Mul(x_normed, op.Cast(self.weight, to=1))
