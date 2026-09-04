# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
# pylint: disable=unused-argument
"""quantized_decomposed ops defined in https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""

from __future__ import annotations

from typing import Optional

from onnxscript import ir
from onnxscript.function_libs.torch_lib.ops import common
from onnxscript.function_libs.torch_lib.registration import torch_op
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType


@torch_op(
    (
        "quantized_decomposed::quantize_per_tensor",
        "quantized_decomposed::quantize_per_tensor.tensor",
        "quantized_decomposed::quantize_per_tensor.tensor2",
    ),
    trace_only=True,
)
def quantized_decomposed_quantize_per_tensor(
    input: TensorType,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
) -> TensorType:
    # TODO(justinchuby): Use dtype when we use opset 21
    return op.QuantizeLinear(input, scale, common.constant(zero_point, dtype=dtype))


@torch_op(
    (
        "quantized_decomposed::dequantize_per_tensor",
        "quantized_decomposed::dequantize_per_tensor.tensor",
        "quantized_decomposed::dequantize_per_tensor.tensor2",
    ),
    trace_only=True,
)
def quantized_decomposed_dequantize_per_tensor(
    input: TensorType,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
    out_dtype: int = -1,
) -> TensorType:
    # TODO(justinchuby): Use dtype when we use opset 21
    dequantized = op.DequantizeLinear(input, scale, common.constant(zero_point, dtype=dtype))
    if out_dtype in (-1, None):
        # out_dtype can be None as well
        return dequantized
    assert out_dtype > 0, f"out_dtype must be -1 or > 0 not {out_dtype}"
    return op.Cast(dequantized, to=out_dtype)


@torch_op(
    (
        "quantized_decomposed::quantize_per_channel",
        "quantized_decomposed::quantize_per_channel.tensor",
        "quantized_decomposed::quantize_per_channel.tensor2",
    ),
    trace_only=True,
)
def quantized_decomposed_quantize_per_channel(
    input: TensorType,
    scales: TensorType,
    zero_points: TensorType,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
) -> TensorType:
    """Affine per channel quantization for the Tensor using the same quantization
    parameters for each channel/axis to map from floating point to quantized values.

    Reference:
    https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py
    ``res = clamp(round(input / scales) + zero_points, quant_min, quant_max)``
    """
    # ONNX QuantizeLinear requires the scale to share the input element type and the
    # zero_point to share the (quantized) output element type. PyTorch passes scales as
    # float64 and zero_points as int64, so cast both to the expected ONNX types.
    scales = op.CastLike(scales, input)
    zero_points = op.Cast(zero_points, to=dtype)
    quantized = op.QuantizeLinear(input, scales, zero_points, axis=axis)
    # QuantizeLinear saturates to the full range of ``dtype``. PyTorch clamps to the
    # explicit ``quant_min``/``quant_max`` instead, so clamp to match its semantics.
    return op.Clip(
        quantized,
        common.constant(quant_min, dtype=dtype),
        common.constant(quant_max, dtype=dtype),
    )


@torch_op(
    (
        "quantized_decomposed::dequantize_per_channel",
        "quantized_decomposed::dequantize_per_channel.tensor",
        "quantized_decomposed::dequantize_per_channel.tensor2",
    ),
    trace_only=True,
)
def quantized_decomposed_dequantize_per_channel(
    input: TensorType,
    scales: TensorType,
    zero_points: Optional[TensorType],
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
    out_dtype: int = -1,
) -> TensorType:
    """Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values.

    Reference:
    https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py
    ``res = (input - zero_points) * scales`` cast to ``out_dtype`` (float32 by default).
    ``quant_min``/``quant_max``/``dtype`` are metadata only and unused in the computation.
    """
    # ONNX DequantizeLinear requires a floating point scale and a zero_point that shares
    # the (quantized) input element type. PyTorch passes scales as float64 and zero_points
    # as int64, so cast scales to float32 (the PyTorch default output type) and zero_points
    # to the quantized input type.
    scales = op.Cast(scales, to=ir.DataType.FLOAT)
    if zero_points is not None:
        zero_points = op.Cast(zero_points, to=dtype)
    dequantized = op.DequantizeLinear(input, scales, zero_points, axis=axis)
    if out_dtype in (-1, None):
        # PyTorch defaults to float32, which DequantizeLinear already produces.
        return dequantized
    assert out_dtype > 0, f"out_dtype must be -1 or > 0 not {out_dtype}"
    return op.Cast(dequantized, to=out_dtype)
