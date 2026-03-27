# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference ir.Function for the CausalConvWithState operator.

Implements depthwise causal N-d convolution with carry state for
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


def causal_conv_nd_with_state(
    *,
    kernel_size: int,
    channels: int,
    ndim: int = 1,
    activation: str = "silu",
) -> ir.Function:
    """Build an ``ir.Function`` for the ``CausalConvWithState`` ONNX operator.

    The Python factory is snake_case (``causal_conv_nd_with_state``) while
    the ir.Function op type string is PascalCase (``"CausalConvWithState"``),
    following the same convention as ``linear_attention`` / ``LinearAttention``.

    Implements depthwise causal N-d convolution with carry state for
    incremental decoding.  The carry state holds the last ``K-1`` positions
    along the **last** (temporal) spatial dimension so autoregressive
    generation avoids recomputing the full causal convolution.

    Supports 1-D, 2-D, and 3-D inputs:

    * 1-D — ``(B, D, T)``         — temporal axis = 2
    * 2-D — ``(B, D, H, W)``      — temporal axis = 3
    * 3-D — ``(B, D, D1, D2, T)`` — temporal axis = 4

    The convolution is always depthwise (``group = channels``).

    .. note:: **Dilation not supported**

       The stateful carry-state mechanism assumes stride=1 and dilation=1.
       Dilated convolutions would require a state buffer of size
       ``dilation*(K-1)`` instead of ``K-1``.

    Args:
        kernel_size: Convolution kernel size ``K``.
        channels: Number of input/output channels ``D`` (always depthwise).
        ndim: Number of spatial dimensions — 1, 2, or 3.
        activation: Fused activation — ``"silu"``, ``"swish"``, or
            ``"none"``.

    Inputs:
        input:      ``(B, D, *spatial)``           — channels-first input
        weight:     ``(D, 1, *[K]*ndim)``          — depthwise weights
        bias:       ``(D,)``                       — bias
        conv_state: ``(B, D, *spatial[:-1], K-1)`` — carry state

    Outputs:
        output:        ``(B, D, *spatial)``           — convolution output
        present_state: ``(B, D, *spatial[:-1], K-1)`` — updated carry state

    .. important::

       The underlying convolution uses **valid** (unpadded) kernels of shape
       ``[K] * ndim`` over all spatial dimensions.  Causal left-padding is
       applied only to the last (temporal) axis, so for ``ndim > 1`` non-
       temporal spatial dimensions shrink by ``K-1``.

       For incremental decoding with carry state, non-temporal spatial
       dimensions are typically size-1, e.g. ``(B, D, 1, T)`` for
       ``ndim=2``, so the output shape effectively matches the input.
    """
    if ndim not in (1, 2, 3):
        raise ValueError(f"ndim must be 1, 2, or 3; got {ndim}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1; got {kernel_size}")
    if channels <= 0:
        raise ValueError(f"channels must be > 0; got {channels}")

    state_width = kernel_size - 1
    # The temporal axis = last spatial dim = index (ndim + 1) in (B, D, *spatial).
    temporal_axis = ndim + 1

    input_val = ir.Value(name="input")
    weight_val = ir.Value(name="weight")
    bias_val = ir.Value(name="bias")
    conv_state_val = ir.Value(name="conv_state")

    graph = ir.Graph(
        inputs=[input_val, weight_val, bias_val, conv_state_val],
        outputs=[],
        nodes=[],
        name="CausalConvWithState_body",
        opset_imports={"": OPSET_VERSION},
    )
    gb = builder.GraphBuilder(graph)
    op = gb.op

    # Step 1: Prepend conv_state along the temporal axis.
    conv_input = op.Concat(conv_state_val, input_val, axis=temporal_axis)

    # Step 2: Extract new carry state — last K-1 positions of conv_input.
    total_len = op.Gather(op.Shape(conv_input), op.Constant(value_int=temporal_axis), axis=0)
    state_start = op.Sub(total_len, op.Constant(value_int=state_width))
    present_state = op.Slice(
        conv_input,
        op.Reshape(state_start, op.Constant(value_ints=[1])),
        op.Reshape(total_len, op.Constant(value_ints=[1])),
        op.Constant(value_ints=[temporal_axis]),
    )
    present_state.name = "present_state"

    # Step 3: Depthwise N-d Conv (group = channels, no padding — already prepended).
    kernel_shape = [kernel_size] * ndim
    dilations = [1] * ndim
    pads = [0] * (2 * ndim)
    conv_out = op.Conv(
        conv_input,
        weight_val,
        kernel_shape=kernel_shape,
        dilations=dilations,
        pads=pads,
        group=channels,
    )

    # Step 4: Add bias — reshape to (1, D, *[1]*ndim) for broadcasting.
    bias_shape = [1, -1] + [1] * ndim
    bias_reshaped = op.Reshape(bias_val, op.Constant(value_ints=bias_shape))
    conv_out = op.Add(conv_out, bias_reshaped)

    # Step 5: Apply activation.
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

    # NOTE: Do not set ``overload`` here — call sites (op.CausalConvWithState)
    # do not set an overload, so setting one on the function would prevent the
    # serializer from matching nodes to this function definition.
    return ir.Function(
        domain=DOMAIN,
        name="CausalConvWithState",
        graph=graph,
        attributes={
            "activation": ir.Attr(
                "activation",
                ir.AttributeType.STRING,
                activation,
            ),
        },
    )


def causal_conv1d_with_state(
    *,
    kernel_size: int,
    channels: int,
    activation: str = "silu",
) -> ir.Function:
    """Build an ``ir.Function`` for 1-D ``CausalConvWithState``.

    .. deprecated::
        Use :func:`causal_conv_nd_with_state` with ``ndim=1`` instead.
        Retained for backward compatibility.
    """
    return causal_conv_nd_with_state(
        kernel_size=kernel_size,
        channels=channels,
        ndim=1,
        activation=activation,
    )


# PascalCase alias — matches the ONNX op type name for discoverability.
CausalConvWithState = causal_conv_nd_with_state
