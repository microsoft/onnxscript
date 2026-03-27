# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference ir.Function for the proposed LinearAttention operator.

Implements the gated delta-rule linear attention recurrence as a
self-contained function: 3D→4D reshape + GQA head expansion + query
scaling + sequential Scan over the time dimension.  The component
(GatedDeltaNet) simply calls
``op.LinearAttention(q, k, v, state, decay, beta)`` — all complexity
lives here.

The gated-delta variant (used by Qwen3.5 GatedDeltaNet) computes:

    S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T k_t)
    o_t = q_t^T S_t

All activations are 3D ``[B, T, H*D]`` (matching the ONNX Attention op
convention).  ``q_num_heads`` and ``kv_num_heads`` attributes tell
the function how to reshape to 4D internally.

The function has 6 required inputs:
  ``(query, key, value, past_state, decay, beta)``

GQA support: when Q/K have fewer heads (q_num_heads) than
V/state (kv_num_heads), the function expands Q/K heads internally
via Tile+Reshape.

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
    q_num_heads: int,
    kv_num_heads: int,
    update_rule: str = "gated_delta",
    scale: float = 1.0,
    stash_type: ir.DataType = ir.DataType.FLOAT,
) -> ir.Function:
    """Build an ir.Function for LinearAttention.

    Inputs (all required):
        query:      (B, T, q_num_heads * d_k)
        key:        (B, T, q_num_heads * d_k)
        value:      (B, T, kv_num_heads * d_v)
        past_state: (B, kv_num_heads, d_k, d_v) — recurrent state
        decay:      (B, T, kv_num_heads * d_k) — per-key-dim decay (log-space);
                    use (B, T, kv_num_heads) for per-head scalar (d_k=1, broadcasts)
        beta:       (B, T, kv_num_heads) — update rate (sigmoid output)

    Outputs:
        output:        (B, T, kv_num_heads * d_v) — attention output (3D)
        present_state: (B, kv_num_heads, d_k, d_v) — updated state

    Args:
        q_num_heads: Number of heads for Q/K (may be fewer than V
            for GQA).  Matches ``kv_num_heads`` in the standard
            Attention op (Q/K are the "key-query" pair here).
        kv_num_heads: Number of heads for V and the output.  Matches
            ``q_num_heads`` in the standard Attention op (V determines
            the output head count).

    .. note:: **Naming convention vs standard Attention**

       In standard GQA (``Attention`` op), Q has *more* heads than K/V,
       so ``q_num_heads >= kv_num_heads``.  In LinearAttention the roles
       are flipped: Q/K are the *key-query* pair (fewer heads) while V
       determines the *output* head count (more heads).  The attribute
       names ``q_num_heads`` and ``kv_num_heads`` therefore map to the
       *opposite* head counts compared to the standard Attention op.
       This matches the ONNX LinearAttention op proposal
       (onnx/onnx#7689) which uses the same naming convention.
        update_rule: One of "linear", "gated", "delta", "gated_delta".
        scale: Scalar multiplier applied to query before the recurrence.
            Per the ONNX LinearAttention op spec, should be set to
            ``1/sqrt(head_dim)`` for correct scaling.
        stash_type: Element type for the Scan body's internal
            computation.  Must match the precision of the inputs
            passed at the call site.  Defaults to ``FLOAT``.

    The function body handles 3D→4D reshape, GQA expansion, query
    scaling, and uses an ONNX Scan op for the sequential recurrence.
    Output is reshaped back to 3D.  A fused kernel can replace the
    entire function for 10-50x speedup.
    """
    valid_rules = ("linear", "gated", "delta", "gated_delta")
    if update_rule not in valid_rules:
        raise ValueError(
            f"Unknown update_rule: {update_rule!r}. Expected one of {valid_rules}."
        )
    if q_num_heads <= 0:
        raise ValueError(f"q_num_heads must be > 0; got {q_num_heads}")
    if kv_num_heads <= 0:
        raise ValueError(f"kv_num_heads must be > 0; got {kv_num_heads}")

    uses_decay = update_rule in ("gated", "gated_delta")
    uses_beta = update_rule in ("delta", "gated_delta")

    # --- Define function inputs (all required) ---
    query = ir.Value(name="query")  # (B, T, q_num_heads * d_k)
    key = ir.Value(name="key")  # (B, T, q_num_heads * d_k)
    value = ir.Value(name="value")  # (B, T, kv_num_heads * d_v)
    past_state = ir.Value(name="past_state")
    decay = ir.Value(name="decay")
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

    # --- Reshape 3D → 4D using head counts ---
    b_dim = op.Shape(query, start=0, end=1)
    t_dim = op.Shape(query, start=1, end=2)

    # Q/K: [B, T, q_num_heads*d_k] → [B, T, q_num_heads, d_k]
    #     → transpose to [B, q_num_heads, T, d_k]
    qk_4d_shape = op.Concat(
        b_dim,
        t_dim,
        op.Constant(value_ints=[q_num_heads, -1]),
        axis=0,
    )
    query_4d = op.Transpose(
        op.Reshape(query, qk_4d_shape), perm=[0, 2, 1, 3]
    )  # [B, q_num_heads, T, d_k]
    key_4d = op.Transpose(
        op.Reshape(key, qk_4d_shape), perm=[0, 2, 1, 3]
    )  # [B, q_num_heads, T, d_k]

    # V: [B, T, kv_num_heads*d_v] → [B, kv_num_heads, T, d_v]
    # Reuse kv_4d_shape for both V and decay (same [B, T, kv_num_heads, -1]).
    kv_4d_shape = op.Concat(
        b_dim,
        t_dim,
        op.Constant(value_ints=[kv_num_heads, -1]),
        axis=0,
    )
    value_4d = op.Transpose(
        op.Reshape(value, kv_4d_shape), perm=[0, 2, 1, 3]
    )  # [B, kv_num_heads, T, d_v]

    # --- GQA: expand Q/K heads to match V head count ---
    if kv_num_heads % q_num_heads != 0:
        raise ValueError(
            f"kv_num_heads ({kv_num_heads}) must be divisible by q_num_heads ({q_num_heads})"
        )
    gqa_ratio = kv_num_heads // q_num_heads
    query_expanded, key_expanded = _expand_kv_heads(op, query_4d, key_4d, gqa_ratio=gqa_ratio)

    # --- Reshape decay/beta 3D → 4D ---
    # decay: (B, T, kv_num_heads * d_k) → (B, T, kv_num_heads, d_k)
    #     → transpose to (B, kv_num_heads, T, d_k)
    decay_4d = op.Transpose(
        op.Reshape(decay, kv_4d_shape), perm=[0, 2, 1, 3]
    )  # [B, kv_num_heads, T, d_k]
    # beta: (B, T, kv_num_heads) → transpose to (B, kv_num_heads, T)
    beta_3d = op.Transpose(beta, perm=[0, 2, 1])  # [B, kv_num_heads, T]

    # --- Apply query scale (matches op spec default of 1/sqrt(d_k)) ---
    # CastLike ensures scale constant matches the input dtype.
    scaled_query = op.Mul(
        query_expanded,
        op.CastLike(op.Constant(value_float=scale), query_expanded),
    )

    # --- Build Scan for sequential recurrence ---
    scan_body = _build_recurrence_body(uses_decay, uses_beta, stash_type=stash_type)

    # Transpose to T-first for Scan: (B, H, T, D) -> (T, B, H, D)
    q_t = op.Transpose(scaled_query, perm=[2, 0, 1, 3])
    k_t = op.Transpose(key_expanded, perm=[2, 0, 1, 3])
    v_t = op.Transpose(value_4d, perm=[2, 0, 1, 3])
    # decay: (B, H, T, d_k) -> (T, B, H, d_k)
    decay_t = op.Transpose(decay_4d, perm=[2, 0, 1, 3])
    # beta: (B, H, T) -> (T, B, H)
    beta_t = op.Transpose(beta_3d, perm=[2, 0, 1])

    present_state, output_t = op.Scan(
        past_state,  # carry: (B, H, d_k, d_v)
        q_t,  # (T, B, H, d_k)
        k_t,  # (T, B, H, d_k)
        v_t,  # (T, B, H, d_v)
        decay_t,  # (T, B, H, d_k)
        beta_t,  # (T, B, H)
        body=scan_body,
        num_scan_inputs=5,
        _outputs=2,
    )
    # present_state: (B, H, d_k, d_v)
    # output_t: (T, B, H, d_v)

    # --- Reshape output 4D → 3D ---
    # (T, B, H, d_v) → (B, T, H, d_v) → (B, T, H*d_v)
    output_bthd = op.Transpose(output_t, perm=[1, 0, 2, 3])
    out_3d_shape = op.Concat(b_dim, t_dim, op.Constant(value_ints=[-1]), axis=0)
    output = op.Reshape(output_bthd, out_3d_shape)  # [B, T, H*d_v]

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
    q_heads_attr = ir.Attr(
        "q_num_heads",
        ir.AttributeType.INT,
        q_num_heads,
        ref_attr_name="q_num_heads",
    )
    kv_heads_attr = ir.Attr(
        "kv_num_heads",
        ir.AttributeType.INT,
        kv_num_heads,
        ref_attr_name="kv_num_heads",
    )
    return ir.Function(
        domain=DOMAIN,
        name="LinearAttention",
        graph=graph,
        attributes={
            "update_rule": update_rule_attr,
            "scale": scale_attr,
            "q_num_heads": q_heads_attr,
            "kv_num_heads": kv_heads_attr,
        },
    )


def _build_recurrence_body(
    uses_decay: bool,
    uses_beta: bool,
    *,
    stash_type: ir.DataType = ir.DataType.FLOAT,
) -> ir.Graph:
    """Build the Scan body for single-token delta-rule recurrence.

    The body operates in ``stash_type`` precision.  Every body input
    carries an explicit ``ir.TensorType`` so that the ONNX serializer
    emits a valid ``type_proto`` — without it ORT cannot infer types
    for the Scan subgraph and the MatMul nodes inside will fail with
    shape-broadcast errors.

    Body inputs (in order):
        1. state: (B, H, d_k, d_v) [carry]
        2. q_t: (B, H, d_k) [scan input]
        3. k_t: (B, H, d_k) [scan input]
        4. v_t: (B, H, d_v) [scan input]
        5. decay_t: (B, H, d_k) [scan input]
        6. beta_t: (B, H) [scan input]

    Body outputs:
        1. new_state: (B, H, d_k, d_v) [carry]
        2. output_t: (B, H, d_v) [scan output]
    """
    batch = ir.SymbolicDim("B")
    dtype = ir.TensorType(stash_type)

    state_in = ir.Value(
        name="state",
        shape=ir.Shape([batch, "H", "d_k", "d_v"]),
        type=dtype,
    )
    q_t = ir.Value(
        name="q_t",
        shape=ir.Shape([batch, "H", "d_k"]),
        type=dtype,
    )
    k_t = ir.Value(
        name="k_t",
        shape=ir.Shape([batch, "H", "d_k"]),
        type=dtype,
    )
    v_t = ir.Value(
        name="v_t",
        shape=ir.Shape([batch, "H", "d_v"]),
        type=dtype,
    )
    decay_t = ir.Value(
        name="decay_t",
        shape=ir.Shape([batch, "H", "d_k"]),
        type=dtype,
    )
    beta_t = ir.Value(
        name="beta_t",
        shape=ir.Shape([batch, "H"]),
        type=dtype,
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
        # decay_t: (B, H, d_k) -> (B, H, d_k, 1) for broadcasting with state (B, H, d_k, d_v)
        g_exp = bop.Exp(bop.Unsqueeze(decay_t, axes_neg1))
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
