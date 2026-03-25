# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference ir.Function for the proposed LinearAttention operator.

Implements the gated delta-rule linear attention recurrence as a
self-contained function: GQA head expansion + sequential Scan over
the time dimension.  The component (GatedDeltaNet) simply calls
``op.LinearAttention(q, k, v, state, decay, beta)`` — all
complexity lives here.

The gated-delta variant (used by Qwen3.5 GatedDeltaNet) computes:

    S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T k_t)
    o_t = q_t^T S_t

GQA support: when Q/K have fewer heads (H_kv) than V/state (H),
the function expands Q/K heads internally via Tile+Reshape.

Op spec: https://github.com/onnx/onnx/issues/7689
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal import builder

from mobius._constants import OPSET_VERSION
from mobius.components._scan_utils import (
    create_body_graph,
    rename_subgraph_values,
)

DOMAIN = "com.microsoft"


def linear_attention(
    *,
    num_k_heads: int,
    num_v_heads: int,
    update_rule: str = "gated_delta",
    scale: float = 1.0,
) -> ir.Function:
    """Build an ir.Function for LinearAttention.

    Inputs:
        query:      (B, H_kv, T, d_k) — query (L2-normalized, unscaled)
        key:        (B, H_kv, T, d_k) — key (L2-normalized)
        value:      (B, H, T, d_v) — value (H >= H_kv)
        past_state: (B, H, d_k, d_v) — recurrent state
        decay:      (B, H, T) — exponential decay gate (log-space)
        beta:       (B, H, T) — update rate (sigmoid output)

    Outputs:
        output:        (B, H, T, d_v) — attention output
        present_state: (B, H, d_k, d_v) — updated recurrent state

    Args:
        num_k_heads: Number of key/query heads (H_kv).
        num_v_heads: Number of value heads (H). Must be >= num_k_heads.
        update_rule: One of "linear", "gated", "delta", "gated_delta".
        scale: Scalar multiplier applied to query before the recurrence.
            Per the ONNX LinearAttention op spec, defaults to
            ``1/sqrt(head_dim)`` in the op proposal.  Callers should
            pass that value explicitly (or 1.0 if queries are
            pre-scaled).

    The function body handles GQA expansion, query scaling, and uses
    an ONNX Scan op for the sequential recurrence.  A fused kernel
    can replace the entire function for 10-50x speedup.
    """
    valid_rules = ("linear", "gated", "delta", "gated_delta")
    if update_rule not in valid_rules:
        raise ValueError(
            f"Unknown update_rule: {update_rule!r}. Expected one of {valid_rules}."
        )

    uses_decay = update_rule in ("gated", "gated_delta")
    uses_beta = update_rule in ("delta", "gated_delta")

    # --- Define function inputs ---
    # query/key/value/past_state/beta accept any floating-point precision
    # (float16/bfloat16/float32) — they form the implicit type parameter T.
    # decay is ALWAYS float32: it is computed in float32 at the call site
    # (HF pattern: -A_log.float().exp() * softplus(a.float() + dt_bias))
    # to avoid exp/softplus overflow in lower precision. Declaring it as FLOAT
    # here keeps it outside the type parameter T so ORT does not see a type
    # conflict when the rest of the inputs are float16/bfloat16.
    # Q/K may have fewer heads (H_kv) than V/state (H) for GQA.
    query = ir.Value(name="query")
    key = ir.Value(name="key")
    value = ir.Value(name="value")
    past_state = ir.Value(name="past_state")
    # decay: (B, H, T) — always float32; Scan body handles unsqueezing.
    decay = ir.Value(name="decay", type=ir.TensorType(ir.DataType.FLOAT))
    # beta: (B, H, T) — same precision as other activations (type T).
    beta = ir.Value(name="beta")
    inputs = [query, key, value, past_state, decay, beta]

    # --- Build function body graph ---
    graph = ir.Graph(
        inputs=inputs,
        outputs=[],
        nodes=[],
        name=f"LinearAttention_{update_rule}_body",
        opset_imports={"": OPSET_VERSION},
    )
    gb = builder.GraphBuilder(graph)
    op = gb.op

    # --- GQA: expand Q/K heads to match V head count ---
    if num_v_heads % num_k_heads != 0:
        raise ValueError(
            f"num_v_heads ({num_v_heads}) must be divisible by num_k_heads ({num_k_heads})"
        )
    gqa_ratio = num_v_heads // num_k_heads
    query_expanded, key_expanded = _expand_kv_heads(op, query, key, gqa_ratio=gqa_ratio)

    # --- Cast to float32 for Scan recurrence precision ---
    # The recurrence requires float32 for numerical stability; inputs may be
    # float16 or bfloat16.  decay is already FLOAT (declared above), so its
    # Cast is always a no-op — kept for uniformity with the other inputs.
    query_f32 = op.Cast(query_expanded, to=ir.DataType.FLOAT)
    key_f32 = op.Cast(key_expanded, to=ir.DataType.FLOAT)
    value_f32 = op.Cast(value, to=ir.DataType.FLOAT)
    state_f32 = op.Cast(past_state, to=ir.DataType.FLOAT)
    decay_f32 = op.Cast(decay, to=ir.DataType.FLOAT)  # no-op: decay is always FLOAT
    beta_f32 = op.Cast(beta, to=ir.DataType.FLOAT)

    # --- Apply query scale (matches op spec default of 1/sqrt(d_k)) ---
    # Scale is applied after f32 cast for precision, and before the Scan
    # recurrence.  A fused kernel reads the scale attribute directly.
    query_f32 = op.Mul(query_f32, op.Constant(value_float=scale))

    # --- Build Scan for sequential recurrence ---
    scan_body = _build_recurrence_body(uses_decay, uses_beta)

    # Transpose to T-first for Scan: (B, H, T, D) -> (T, B, H, D)
    q_t = op.Transpose(query_f32, perm=[2, 0, 1, 3])
    k_t = op.Transpose(key_f32, perm=[2, 0, 1, 3])
    v_t = op.Transpose(value_f32, perm=[2, 0, 1, 3])
    # decay/beta: (B, H, T) -> (T, B, H)
    decay_t = op.Transpose(decay_f32, perm=[2, 0, 1])
    beta_t = op.Transpose(beta_f32, perm=[2, 0, 1])

    present_state, output_t = op.Scan(
        state_f32,  # carry: (B, H, d_k, d_v)
        q_t,  # (T, B, H, d_k)
        k_t,  # (T, B, H, d_k)
        v_t,  # (T, B, H, d_v)
        decay_t,  # (T, B, H)
        beta_t,  # (T, B, H)
        body=scan_body,
        num_scan_inputs=5,
        _outputs=2,
    )
    # present_state: (B, H, d_k, d_v)
    # output_t: (T, B, H, d_v)

    # Transpose output back: (T, B, H, d_v) -> (B, H, T, d_v)
    output = op.Transpose(output_t, perm=[1, 2, 0, 3])

    # Cast outputs back to the caller's input precision (no-op for float32)
    output = op.CastLike(output, query)
    present_state = op.CastLike(present_state, query)

    output.name = "output"
    present_state.name = "present_state"
    graph.outputs.extend([output, present_state])

    # --- Build the ir.Function ---
    update_rule_attr = ir.Attr(
        "update_rule",
        ir.AttributeType.STRING,
        update_rule,
        ref_attr_name="update_rule",
    )
    scale_attr = ir.Attr(
        "scale",
        ir.AttributeType.FLOAT,
        scale,
        ref_attr_name="scale",
    )
    return ir.Function(
        domain=DOMAIN,
        name="LinearAttention",
        graph=graph,
        attributes={
            "update_rule": update_rule_attr,
            "scale": scale_attr,
        },
    )


def _build_recurrence_body(
    uses_decay: bool,
    uses_beta: bool,
) -> ir.Graph:
    """Build the Scan body for single-token delta-rule recurrence.

    Body inputs (in order):
        1. state: (B, H, d_k, d_v) [carry]
        2. q_t: (B, H, d_k) [scan input]
        3. k_t: (B, H, d_k) [scan input]
        4. v_t: (B, H, d_v) [scan input]
        5. decay_t: (B, H) [scan input]
        6. beta_t: (B, H) [scan input]

    Body outputs:
        1. new_state: (B, H, d_k, d_v) [carry]
        2. output_t: (B, H, d_v) [scan output]
    """
    batch = ir.SymbolicDim("B")

    state_in = ir.Value(
        name="state",
        shape=ir.Shape([batch, "H", "d_k", "d_v"]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    q_t = ir.Value(
        name="q_t",
        shape=ir.Shape([batch, "H", "d_k"]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    k_t = ir.Value(
        name="k_t",
        shape=ir.Shape([batch, "H", "d_k"]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    v_t = ir.Value(
        name="v_t",
        shape=ir.Shape([batch, "H", "d_v"]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    decay_t = ir.Value(
        name="decay_t",
        shape=ir.Shape([batch, "H"]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    beta_t = ir.Value(
        name="beta_t",
        shape=ir.Shape([batch, "H"]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )

    body_graph, body_builder = create_body_graph(
        state_inputs=[state_in],
        scan_inputs=[q_t, k_t, v_t, decay_t, beta_t],
        name="delta_recurrence",
    )
    bop = body_builder.op

    # Shared axes constants (deduplicated)
    axes_neg2 = bop.Constant(value_ints=[-2])
    axes_neg1 = bop.Constant(value_ints=[-1])

    # --- State decay: state = exp(g) * past_state ---
    if uses_decay:
        # decay_t: (B, H) -> (B, H, 1, 1)
        axes_23 = bop.Constant(value_ints=[2, 3])
        g_exp = bop.Exp(bop.Unsqueeze(decay_t, axes_23))
        state = bop.Mul(state_in, g_exp)
    else:
        state = state_in

    # --- Retrieval: k @ state -> (B, H, d_v) ---
    # k_row: (B, H, 1, d_k) @ state: (B, H, d_k, d_v) -> (B, H, 1, d_v)
    k_row = bop.Unsqueeze(k_t, axes_neg2)
    retrieval = bop.Squeeze(bop.MatMul(k_row, state), axes_neg2)

    # --- State update ---
    if uses_beta:
        # delta = (v - retrieval) * beta
        delta = bop.Sub(v_t, retrieval)
        beta_expanded = bop.Unsqueeze(beta_t, axes_neg1)  # (B, H, 1)
        delta = bop.Mul(delta, beta_expanded)
    else:
        delta = v_t

    # Outer product: k^T @ delta -> (B, H, d_k, d_v)
    k_col = bop.Unsqueeze(k_t, axes_neg1)  # (B, H, d_k, 1)
    delta_row = bop.Unsqueeze(delta, axes_neg2)  # (B, H, 1, d_v)
    outer = bop.MatMul(k_col, delta_row)
    new_state = bop.Add(state, outer)
    new_state.name = "new_state"

    # --- Output: q @ new_state -> (B, H, d_v) ---
    q_row = bop.Unsqueeze(q_t, axes_neg2)  # (B, H, 1, d_k)
    output_t = bop.Squeeze(bop.MatMul(q_row, new_state), axes_neg2)
    output_t.name = "output_t"

    body_graph.outputs.extend([new_state, output_t])
    rename_subgraph_values(body_graph, "dn_")

    return body_graph


def _expand_kv_heads(op, query, key, *, gqa_ratio: int):
    """Expand Q/K heads to match V head count for GQA.

    When gqa_ratio > 1, each Q/K head is tiled ``gqa_ratio`` times along
    a new dim and then reshaped to merge heads.
    When gqa_ratio == 1, this is a no-op (returns inputs unchanged).

    The expansion ratio is computed at graph-build time so the Tile
    repeats are a static Constant — this avoids Shape→Gather→Div ops
    whose int64 outputs cause CUDA EP memory placement issues.

    Args:
        op: ONNX op builder.
        query: (B, H_kv, T, d_k)
        key: (B, H_kv, T, d_k)
        gqa_ratio: ``num_v_heads // num_k_heads`` (computed at build time).

    Returns:
        query: (B, H, T, d_k)
        key: (B, H, T, d_k)
    """
    if gqa_ratio == 1:
        return query, key

    # (B, H_kv, T, d_k) -> (B, H_kv, 1, T, d_k)
    axes_2 = op.Constant(value_ints=[2])
    q_5d = op.Unsqueeze(query, axes_2)
    k_5d = op.Unsqueeze(key, axes_2)

    # Tile along dim 2 by the static ratio
    repeat_vec = op.Constant(value_ints=[1, 1, gqa_ratio, 1, 1])
    q_tiled = op.Tile(q_5d, repeat_vec)
    k_tiled = op.Tile(k_5d, repeat_vec)

    # Reshape: (B, H_kv, ratio, T, d_k) -> (B, H_kv*ratio, T, d_k)
    # Extract B, T, d_k from the original 4D query to build the target shape.
    # We cannot use '0' sentinels because ONNX Reshape copies from the same
    # positional index, and the 5D→4D dimension mapping doesn't align.
    b_dim = op.Shape(query, start=0, end=1)
    t_dim = op.Shape(query, start=2, end=3)
    dk_dim = op.Shape(query, start=3, end=4)
    expanded_shape = op.Concat(
        b_dim,
        op.Constant(value_ints=[-1]),
        t_dim,
        dk_dim,
        axis=0,
    )
    return op.Reshape(q_tiled, expanded_shape), op.Reshape(k_tiled, expanded_shape)
