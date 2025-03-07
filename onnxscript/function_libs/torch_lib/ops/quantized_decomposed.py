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
