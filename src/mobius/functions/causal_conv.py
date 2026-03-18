# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference ir.Function for the proposed CausalConv1DWithState operator.

Implements depthwise causal 1D convolution with carry state for
incremental decoding.  During autoregressive generation, the carry state
(last K-1 time steps) avoids recomputing the full causal convolution.

Op spec: https://github.com/onnx/onnx/issues/7689
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal import builder

from mobius._constants import OPSET_VERSION

DOMAIN = "com.microsoft"


# TODO(justinchuby): Simplify function creation boilerplate


def causal_conv1d_with_state(
    *,
    kernel_size: int,
    activation: str = "silu",
) -> ir.Function:
    """Build an ir.Function for CausalConv1DWithState.

    A single function definition works for all channel sizes because the
    Conv ``group`` attribute is declared as a function-level attribute via
    ``ref_attr_name`` and supplied by the caller at each call site.

    Args:
        kernel_size: Convolution kernel size (K).
        activation: Fused activation — ``"silu"``, ``"swish"``, or
            ``"none"``.

    Function attributes:
        group: INT — number of groups for depthwise Conv (= channels D).
            Passed by the caller and forwarded to the inner Conv node.

    Inputs:
        input:      (B, D, T)   — channels-first input
        weight:     (D, 1, K)   — depthwise conv weights
        bias:       (D,)        — bias (pass zeros if not needed)
        conv_state: (B, D, K-1) — carry state from previous step

    Outputs:
        output:        (B, D, T)   — convolution output
        present_state: (B, D, K-1) — updated carry state

    The function body implements:
        1. Prepend conv_state to input along the time axis
        2. Slice out the new carry state (last K-1 positions)
        3. Apply depthwise Conv1d (group from function attr, no padding)
        4. Add bias
        5. Apply activation (SiLU by default)
    """
    state_width = kernel_size - 1

    # --- Define function inputs (shapes/types left dynamic) ---
    input_val = ir.Value(name="input")
    weight_val = ir.Value(name="weight")
    bias_val = ir.Value(name="bias")
    conv_state_val = ir.Value(name="conv_state")

    # --- Build function body graph ---
    graph = ir.Graph(
        inputs=[input_val, weight_val, bias_val, conv_state_val],
        outputs=[],
        nodes=[],
        name="CausalConv1DWithState_body",
        opset_imports={"": OPSET_VERSION},
    )
    gb = builder.GraphBuilder(graph)
    op = gb.op

    # Step 1: Prepend conv_state along time axis
    # conv_input: (B, D, K-1 + T)
    conv_input = op.Concat(conv_state_val, input_val, axis=2)

    # Step 2: Extract new carry state — last K-1 positions of conv_input
    total_len = op.Gather(op.Shape(conv_input), op.Constant(value_int=2), axis=0)
    state_start = op.Sub(total_len, op.Constant(value_int=state_width))
    present_state = op.Slice(
        conv_input,
        op.Reshape(state_start, op.Constant(value_ints=[1])),
        op.Reshape(total_len, op.Constant(value_ints=[1])),
        op.Constant(value_ints=[2]),
    )
    present_state.name = "present_state"

    # Step 3: Depthwise Conv1d (group forwarded from function attribute)
    # ir.RefAttr makes the Conv node's ``group`` reference the
    # function-level ``group`` attribute supplied by each caller.
    conv_out = op.Conv(
        conv_input,
        weight_val,
        group=ir.RefAttr("group", "group", ir.AttributeType.INT),
        pads=[0, 0],
    )

    # Step 4: Add bias — reshape bias to (1, D, 1) for broadcasting
    bias_reshaped = op.Reshape(bias_val, op.Constant(value_ints=[1, -1, 1]))
    conv_out = op.Add(conv_out, bias_reshaped)

    # Step 5: Apply activation
    if activation in ("silu", "swish"):
        output = op.Mul(conv_out, op.Sigmoid(conv_out))
    elif activation == "none":
        output = conv_out
    else:
        raise ValueError(
            f"Unsupported activation: {activation!r}. Expected 'silu', 'swish', or 'none'."
        )
    output.name = "output"

    graph.outputs.extend([output, present_state])

    # --- Build the ir.Function ---
    # Function signature declares attributes normally.
    # Inner Conv node references 'group' via ir.RefAttr.
    return ir.Function(
        domain=DOMAIN,
        name="CausalConv1DWithState",
        overload=activation,
        graph=graph,
        attributes={
            "activation": ir.Attr(
                "activation",
                ir.AttributeType.STRING,
                activation,
            ),
            "group": ir.Attr(
                "group",
                ir.AttributeType.INT,
                0,
            ),
        },
    )
