# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
"""torch.ops.aten operators under the `nn` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""

# pylint: disable=unused-argument

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, TypeVar, Union

import onnx

from onnxscript import BFLOAT16, BOOL, DOUBLE, FLOAT, FLOAT16, INT64
from onnxscript.function_libs.torch_lib.ops import common as common_ops
from onnxscript.function_libs.torch_lib.registration import torch_op
from onnxscript.function_libs.torch_lib.tensor_typing import (
    IntType,
    TFloat,
    TFloatOrBFloat16,
    TFloatOrUInt8,
    TReal,
    TTensor,
)
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType

_MATH_PI = math.pi
Rank = common_ops.Rank

# All float types but float32
TFloatUnlessFloat32 = TypeVar("TFloatUnlessFloat32", bound=Union[BFLOAT16, FLOAT16, DOUBLE])


@torch_op("aten::adaptive_avg_pool1d", traceable=True)
def aten_adaptive_avg_pool1d(self: TFloat, output_size: INT64[1]) -> TFloat:
    """adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor"""

    # assert output_size == [1]
    # TODO(justinchuby): Specify input constraints

    if Rank(self) == 2:
        # Unbatched case
        self = op.Unsqueeze(self, op.Constant(value_ints=[0]))
        pooled = op.GlobalAveragePool(self)
        result = op.Squeeze(pooled, op.Constant(value_ints=[0]))
    else:
        result = op.GlobalAveragePool(self)

    return result


@torch_op("aten::adaptive_avg_pool2d", traceable=True)
def aten_adaptive_avg_pool2d(self: TFloat, output_size: INT64[2]) -> TFloat:
    """adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor"""

    # assert output_size == [1, 1]
    # TODO(justinchuby): Specify input constraints

    if Rank(self) == 3:
        # Unbatched case
        self = op.Unsqueeze(self, op.Constant(value_ints=[0]))
        pooled = op.GlobalAveragePool(self)
        result = op.Squeeze(pooled, op.Constant(value_ints=[0]))
    else:
        result = op.GlobalAveragePool(self)

    return result


@torch_op("aten::adaptive_avg_pool3d", traceable=True)
def aten_adaptive_avg_pool3d(self: TFloat, output_size: INT64[3]) -> TFloat:
    """adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor"""

    # assert output_size == [1, 1, 1]
    # TODO(justinchuby): Specify input constraints

    if Rank(self) == 4:
        # Unbatched case
        self = op.Unsqueeze(self, op.Constant(value_ints=[0]))
        pooled = op.GlobalAveragePool(self)
        result = op.Squeeze(pooled, op.Constant(value_ints=[0]))
    else:
        result = op.GlobalAveragePool(self)

    return result


def aten_adaptive_max_pool1d(
    self: TensorType, output_size: Sequence[int]
) -> tuple[TensorType, TensorType]:
    """adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_adaptive_max_pool2d(
    self: TensorType, output_size: Sequence[int]
) -> tuple[TensorType, TensorType]:
    """adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_adaptive_max_pool2d_backward(
    grad_output: TensorType, self: TensorType, indices: TensorType
) -> TensorType:
    """adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor"""

    raise NotImplementedError()


def aten_adaptive_max_pool3d(
    self: TensorType, output_size: Sequence[int]
) -> tuple[TensorType, TensorType]:
    """adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_adaptive_max_pool3d_backward(
    grad_output: TensorType, self: TensorType, indices: TensorType
) -> TensorType:
    """adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor"""

    raise NotImplementedError()


def _adjust_attributes_of_avg_pool(
    expand_size: int,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """Adjust attributes of avg_pool to match ONNX specification."""

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2
    elif len(padding) == 2:
        pads = padding * expand_size
    else:
        pads = padding * 2

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride

    return (kernel_shape, strides, pads)


@torch_op("aten::avg_pool1d", trace_only=True)
def aten_avg_pool1d(
    self: TFloat,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0,),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> TFloat:
    """avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor"""

    # Torch prefer to use single number x for kerne,stride,pad,dilation on both side implicitly
    # But ONNX needs pair number [x,y] to specify on each side explicitly
    # For pool3d, this number should be 3
    expand_size = 1

    kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
        expand_size, kernel_size, stride, padding
    )

    result = op.AveragePool(
        self,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
    )

    return result


@torch_op("aten::avg_pool2d", trace_only=True)
def aten_avg_pool2d(
    self: TFloat,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> TFloat:
    """avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor"""

    # Torch prefer to use single number x for kerne,stride,pad,dilation on both side implicitly
    # But ONNX needs pair number [x,y] to specify on each side explicitly
    # For pool3d, this number should be 3
    expand_size = 2

    kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
        expand_size, kernel_size, stride, padding
    )

    result = op.AveragePool(
        self,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
    )

    # TODO: if want to support divisor_override argument, need to op.Mul(result, mask)
    # mask = [
    #    1, 2, 3, S,..3, 2, 1
    #    2, 4, 6, 2S, 6, 4, 2
    #    3, 6, 9, 3S, 9, 6, 3
    #    S, 2S,3S,SS,3S,2S, S
    #    3, 6, 9, 3S, 9, 6, 3
    #    2, 4, 6, 2S, 6, 4, 2
    #    1, 2, 3, S,..3, 2, 1
    # ]
    # S is stride size, in this case S=4,
    # S may dup lot of times according to the image size

    return result


def aten_avg_pool2d_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
) -> TensorType:
    """avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::avg_pool3d", trace_only=True)
def aten_avg_pool3d(
    self: TFloat,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> TFloat:
    """avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor"""

    # Torch prefer to use single number x for kerne,stride,pad,dilation on both side implicitly
    # But ONNX needs pair number [x,y] to specify on each side explicitly
    # For pool3d, this number should be 3
    expand_size = 3

    kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
        expand_size, kernel_size, stride, padding
    )

    result = op.AveragePool(
        self,
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
    )

    # TODO: if want to support divisor_override argument, need to op.Mul(result, mask)
    # mask = [
    #    1, 2, 3, S,..3, 2, 1
    #    2, 4, 6, 2S, 6, 4, 2
    #    3, 6, 9, 3S, 9, 6, 3
    #    S, 2S,3S,SS,3S,2S, S
    #    3, 6, 9, 3S, 9, 6, 3
    #    2, 4, 6, 2S, 6, 4, 2
    #    1, 2, 3, S,..3, 2, 1
    # ]
    # S is stride size, in this case S=4,
    # S may dup lot of times according to the image size

    return result


def aten_avg_pool3d_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
) -> TensorType:
    """avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor"""

    raise NotImplementedError()


def aten_binary_cross_entropy(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
) -> TensorType:
    """binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


def aten_binary_cross_entropy_backward(
    grad_output: TensorType,
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
) -> TensorType:
    """binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::celu")
def aten_celu(self: FLOAT, alpha: float = 1.0) -> FLOAT:
    """celu(Tensor self, Scalar alpha=1.0) -> Tensor"""

    return op.Celu(self, alpha=alpha)  # op.Celu only support float32


@torch_op("aten::celu", traceable=True)
def aten_celu_type_promoted(
    self: TFloatUnlessFloat32, alpha: float = 1.0
) -> TFloatUnlessFloat32:
    """celu(Tensor self, Scalar alpha=1.0) -> Tensor"""

    self_upcasted = op.Cast(self, to=FLOAT.dtype)
    return op.CastLike(op.Celu(self_upcasted, alpha=alpha), self)


@torch_op("aten::col2im", trace_only=True)
def aten_col2im(
    self: TReal,
    output_size: INT64,
    kernel_size: INT64,
    dilation: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    stride: Sequence[int] = (1, 1),
) -> TReal:
    """col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor"""

    # assert(len(output_size)==2) for ONNX
    # assert(len(kernel_size)==2) for ONNX
    # assert(len(dilation)==2) for ONNX
    # assert(len(stride)==2) for ONNX

    # The pads should be [w, x, y, z] for ONNX
    if len(padding) == 1:  # [w] -> [w, w, w, w]
        pads = padding * 4
    elif len(padding) == 2:  # [w, x] -> [w, x, w, x]
        pads = padding * 2
    else:  # assert len(padding) == 4, already [w, x, y, z]
        pads = padding

    # Only one ONNX op here so didn't write a private function
    return op.Col2Im(
        self,
        output_size,
        kernel_size,
        dilations=dilation,
        pads=pads,
        strides=stride,
    )


def aten_conv_depthwise3d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: INT64,
    dilation: Sequence[int],
) -> TensorType:
    """conv_depthwise3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, SymInt[3] padding, int[3] dilation) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::cross_entropy_loss", traceable=True)
def aten_cross_entropy_loss(
    self: TFloatOrBFloat16,
    target: IntType,
    weight: Optional[TFloatOrBFloat16] = None,
    reduction: int = 1,  # default is 'mean'
    ignore_index: int = -100,
    label_smoothing: float = 0.0,  # this was ignored due to ONNX not support
) -> TFloatOrBFloat16:
    """cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, float label_smoothing=0.0) -> Tensor"""

    if reduction == 0:  # "none"
        result, _ = op.SoftmaxCrossEntropyLoss(
            self, target, weight, reduction="none", ignore_index=ignore_index
        )
    elif reduction == 2:  # "sum"
        result, _ = op.SoftmaxCrossEntropyLoss(
            self, target, weight, reduction="sum", ignore_index=ignore_index
        )
    else:  # "mean", default
        result, _ = op.SoftmaxCrossEntropyLoss(
            self, target, weight, reduction="mean", ignore_index=ignore_index
        )

    return result


@torch_op("aten::elu")
def aten_elu(
    self: TFloat,
    alpha: float = 1.0,
    scale: float = 1.0,
    input_scale: float = 1.0,
) -> TFloat:
    """elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor"""

    # del scale
    # del input_scale
    return op.Elu(self, alpha=alpha)


def aten_elu_backward(
    grad_output: TensorType,
    alpha: float,
    scale: float,
    input_scale: float,
    is_result: bool,
    self_or_result: TensorType,
) -> TensorType:
    """elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor"""

    raise NotImplementedError()


def aten_flatten_dense_tensors(tensors: Sequence[TensorType]) -> TensorType:
    """flatten_dense_tensors(Tensor[] tensors) -> Tensor"""

    raise NotImplementedError()


def aten_fractional_max_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    random_samples: TensorType,
) -> tuple[TensorType, TensorType]:
    """fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_fractional_max_pool2d_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    indices: TensorType,
) -> TensorType:
    """fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor"""

    raise NotImplementedError()


def aten_fractional_max_pool3d(
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    random_samples: TensorType,
) -> tuple[TensorType, TensorType]:
    """fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_fractional_max_pool3d_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    indices: TensorType,
) -> TensorType:
    """fractional_max_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::gelu", trace_only=True)
def aten_gelu(self: TReal, approximate: str = "none") -> TReal:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""

    if approximate == "tanh":
        result = _aten_gelu_approximate_tanh(self)
    else:
        result = _aten_gelu_approximate_none(self)
    return result


@torch_op("aten::gelu", private=True)
def _aten_gelu_approximate_none(self: TReal) -> TReal:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""

    # GELU(x) = 0.5 * x * [1 + ERF(x/sqrt(2)]
    inner = op.Div(self, 1.4142135623730951)
    erf = op.Erf(inner)
    inner = op.Add(erf, 1)
    inner = op.Mul(self, inner)
    result = op.Mul(0.5, inner)
    return result


@torch_op("aten::gelu", private=True)
def _aten_gelu_approximate_tanh(self: TReal) -> TReal:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""

    # GELU(x) = 0.5 * x * {1 + Tanh[\sqrt(2/pi) * (x + 0.044715 * x^3)]}
    cubed = op.Pow(self, 3)
    inner = op.Mul(0.044715, cubed)
    inner = op.Add(self, inner)
    # Prefer explicit graph construction over precomputed constants for clarity.
    two_over_pi = op.CastLike(op.Div(2.0, _MATH_PI), self)
    inner = op.Mul(op.Sqrt(two_over_pi), inner)
    inner = op.Tanh(inner)
    inner = op.Add(inner, 1)
    inner = op.Mul(self, inner)
    result = op.Mul(0.5, inner)
    return result


def aten_gelu_backward(
    grad_output: TensorType, self: TensorType, approximate: str = "none"
) -> TensorType:
    """gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::glu", traceable=True)
def aten_glu(self: TFloat, dim: int = -1) -> TFloat:
    """glu(Tensor self, int dim=-1) -> Tensor"""

    first, second = op.Split(self, axis=dim, num_outputs=2)
    result = op.Mul(first, op.Sigmoid(second))
    return result


def aten_glu_backward(grad_output: TensorType, self: TensorType, dim: int) -> TensorType:
    """glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor"""

    raise NotImplementedError()


def aten_glu_backward_jvp(
    grad_x: TensorType,
    grad_glu: TensorType,
    x: TensorType,
    dgrad_glu: TensorType,
    dx: TensorType,
    dim: int,
) -> TensorType:
    """glu_backward_jvp(Tensor grad_x, Tensor grad_glu, Tensor x, Tensor dgrad_glu, Tensor dx, int dim) -> Tensor"""

    raise NotImplementedError()


def aten_glu_jvp(glu: TensorType, x: TensorType, dx: TensorType, dim: int) -> TensorType:
    """glu_jvp(Tensor glu, Tensor x, Tensor dx, int dim) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::hardsigmoid")
def aten_hardsigmoid(self: TFloat) -> TFloat:
    """hardsigmoid(Tensor self) -> Tensor"""

    return op.HardSigmoid(self, alpha=1 / 6, beta=1 / 2)


def aten_hardsigmoid_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    """hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::hardswish")
def aten_hardswish(self: TFloat) -> TFloat:
    """hardswish(Tensor self) -> Tensor"""

    return op.HardSwish(self)


def aten_hardswish_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    """hardswish_backward(Tensor grad_output, Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::hardtanh")
def aten_hardtanh(self: TReal, min_val: float = -1.0, max_val: float = 1.0) -> TReal:
    """hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor"""

    return op.Clip(self, min_val, max_val)


def aten_hardtanh_backward(
    grad_output: TensorType, self: TensorType, min_val: float, max_val: float
) -> TensorType:
    """hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor"""

    raise NotImplementedError()


def aten_huber_loss(
    self: TensorType, target: TensorType, reduction: int = 1, delta: float = 1.0
) -> TensorType:
    """huber_loss(Tensor self, Tensor target, int reduction=Mean, float delta=1.0) -> Tensor"""

    raise NotImplementedError()


def aten_huber_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int, delta: float
) -> TensorType:
    """huber_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta) -> Tensor"""

    raise NotImplementedError()


def aten_im2col(
    self: TensorType,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
) -> TensorType:
    """im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor"""

    raise NotImplementedError()


def aten_infinitely_differentiable_gelu_backward(
    grad: TensorType, self: TensorType
) -> TensorType:
    """infinitely_differentiable_gelu_backward(Tensor grad, Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_l1_loss(self: TensorType, target: TensorType, reduction: int = 1) -> TensorType:
    """l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::leaky_relu")
def aten_leaky_relu(self: TFloatOrBFloat16, negative_slope: float = 0.01) -> TFloatOrBFloat16:
    """leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor"""

    return op.LeakyRelu(self, alpha=negative_slope)


def aten_leaky_relu_backward(
    grad_output: TensorType, self: TensorType, negative_slope: float, self_is_result: bool
) -> TensorType:
    """leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::linear")
def aten_linear(input: TFloat, weight: TFloat) -> TFloat:
    """linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"""

    # NOTE: The symbolic function in torch.onnx also uses Gemm in certain cases
    # Optimizers may consider this path and replace it with Gemm
    # We do not use Gemm here because input can have batch dimensions, which Gemm does not support
    weight_transposed = op.Transpose(weight, perm=[1, 0])
    return op.MatMul(input, weight_transposed)


@torch_op("aten::linear")
def aten_linear_bias(input: TFloat, weight: TFloat, bias: TFloat) -> TFloat:
    """linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"""

    # NOTE: The symbolic function in torch.onnx also uses Gemm in certain cases
    # Optimizers may consider this path and replace it with Gemm
    # We do not use Gemm here because input can have batch dimensions, which Gemm does not support
    weight_transposed = op.Transpose(weight, perm=[1, 0])
    mul = op.MatMul(input, weight_transposed)
    return op.Add(mul, bias)


@torch_op("aten::log_sigmoid")
def aten_log_sigmoid(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """log_sigmoid(Tensor self) -> Tensor"""

    return op.Log(op.Sigmoid(self))


def aten_log_sigmoid_backward(
    grad_output: TensorType, self: TensorType, buffer: TensorType
) -> TensorType:
    """log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor"""

    raise NotImplementedError()


def aten_log_sigmoid_forward(self: TensorType) -> tuple[TensorType, TensorType]:
    """log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)"""

    raise NotImplementedError()


def aten_logit_backward(
    grad_output: TensorType, self: TensorType, eps: Optional[float] = None
) -> TensorType:
    """logit_backward(Tensor grad_output, Tensor self, float? eps=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::max_pool1d", trace_only=True)
def aten_max_pool1d(
    self: TFloatOrUInt8,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    ceil_mode: bool = False,
) -> TFloatOrUInt8:
    """max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a tuple of three ints for all sides explicitly.
    expand_size = 1

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(self, kernel_shape, strides, pads, dilations, ceil_mode, 2)


@torch_op("aten::max_pool1d_with_indices", trace_only=True)
def aten_max_pool1d_with_indices(
    self: TFloatOrUInt8,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    ceil_mode: bool = False,
) -> Tuple[TFloatOrUInt8, INT64]:
    """max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a tuple of three ints for all sides explicitly.
    expand_size = 1

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_with_indices_onnx(
        self,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        2,
        ([1] * expand_size),
        ([0] * expand_size),
        ([2 + i for i in range(expand_size)]),
    )


def _adjust_attributes_of_max_pool(
    expand_size: int,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    if isinstance(dilation, int):
        dilations = [dilation] * expand_size
    else:
        dilations = dilation

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size

    # NOTE: expand_size is the dimension of pooling kernel,
    # ONNX needs begin and end padding so we need to double the padding

    # NOTE: expand size prevents padding from being a single int in
    # multiple dimension cases
    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2
    elif len(padding) == 2:
        # 2D padding
        pads = padding * 2
    elif len(padding) == 3:
        # 3D padding
        pads = padding * 2
    else:
        # When padding is already done for all dimensions,
        # we don't need to double it
        # eg: (1, 1, 1, 1, 1, 1)
        pads = padding

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride

    return (kernel_shape, strides, pads, dilations)


@torch_op("aten::max_pool2d", trace_only=True)
def aten_max_pool2d(
    self: TFloatOrUInt8,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> TFloatOrUInt8:
    """max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a pair of number [x,y] on each side explicitly.
    expand_size = 2

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(self, kernel_shape, strides, pads, dilations, ceil_mode, 3)


@torch_op("internal::max_pool", private=True, traceable=True)
def _aten_max_pool_onnx(
    self: TFloatOrUInt8,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
) -> TFloatOrUInt8:
    self_rank_is_unbatched_rank = Rank(self) == unbatched_rank
    if self_rank_is_unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = op.Unsqueeze(self, op.Constant(value_ints=[0]))

    pool_result, _ = op.MaxPool(
        self,
        ceil_mode=ceil_mode,
        dilations=dilations,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
    )

    if self_rank_is_unbatched_rank:
        pool_result = op.Squeeze(pool_result, op.Constant(value_ints=[0]))

    return pool_result


@torch_op("aten::max_pool3d", trace_only=True)
def aten_max_pool3d(
    self: TFloatOrUInt8,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    ceil_mode: bool = False,
) -> TFloatOrUInt8:
    """max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a tuple of three ints for all sides explicitly.
    expand_size = 3

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(self, kernel_shape, strides, pads, dilations, ceil_mode, 4)


@torch_op("aten::max_pool2d_with_indices", trace_only=True)
def aten_max_pool2d_with_indices(
    self: TFloatOrUInt8,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> Tuple[TFloatOrUInt8, INT64]:
    """max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a pair of number [x,y] on each side explicitly.
    expand_size = 2

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_with_indices_onnx(
        self,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        3,
        ([1] * expand_size),
        ([0] * expand_size),
        ([2 + i for i in range(expand_size)]),
    )


def aten_max_pool2d_with_indices_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
    indices: TensorType,
) -> TensorType:
    """max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::max_pool3d_with_indices", trace_only=True)
def aten_max_pool3d_with_indices(
    self: TFloatOrUInt8,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    ceil_mode: bool = False,
) -> Tuple[TFloatOrUInt8, INT64]:
    """max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)"""

    # Torch prefers to use single number x for kernel, stride, pad and dilation on both sides implicitly.
    # But ONNX needs to specify a tuple of three ints for all sides explicitly.
    expand_size = 3

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_with_indices_onnx(
        self,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        4,
        ([1] * expand_size),
        ([0] * expand_size),
        ([2 + i for i in range(expand_size)]),
    )


@torch_op("internal::max_pool_with_indices", private=True, traceable=True)
def _aten_max_pool_with_indices_onnx(
    self: TFloatOrUInt8,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
    n_dims_one: Sequence[int],
    n_dims_zero: Sequence[int],
    n_dims_axes: Sequence[int],
) -> Tuple[TFloatOrUInt8, INT64]:
    self_rank_is_unbatched_rank = Rank(self) == unbatched_rank
    if self_rank_is_unbatched_rank:
        self = op.Unsqueeze(self, axes=0)

    pool_result, indices = op.MaxPool(
        self,
        ceil_mode=ceil_mode,
        dilations=dilation,
        kernel_shape=kernel_size,
        pads=padding,
        strides=stride,
    )

    # Simple but hacky way to get flattened indices values
    # to be used to convert the indices values to non-flattened.
    # In ONNX the indices are computed as a flatten 1-D tensor,
    # so the values in indices are in [0, N x C x D1 x ... x Dn).
    # To convert the indices to the same format used by PyTorch,
    # we first execute a maxpool with a kernel and stride of 1 on the same input.
    # This will result in a tensor of indices in which each index will have it's own value.
    # Using this tensor as a reference, we extract the first index of each axis and subtract
    # it from each index of this axis in the indices to convert.
    # This step will result in a tensor where each dimension has values of indices within
    # the dimension it is in.
    # For Maxpool1d(kernel=1,stride=1,return_indices=True), with the input torch.ones(1,2,2).
    # The computed indices are the following:
    # output indices pytorch :
    #     [[0,1],
    #      [0,1]]
    # output indices onnx:
    #     [[0,1],
    #      [2,3]]
    # The purpose was to convert the indices from one format to the other to be able to match the results.
    # So flattened_indices will have the value of each index and will be equal to :
    #     [[0,1],
    #     [2,3]]
    # Then call Slice to get the first value of each line (so 0 and 2).
    # And the subtraction executes :
    #     [[0-0,1-0],
    #     [2-2,3-2]]
    # So indices results to the expected output which is :
    #     [[0,1],
    #     [0,1]]
    # For more information :
    # https://github.com/pytorch/pytorch/pull/16455#issuecomment-460776407
    _, flatten_indices = op.MaxPool(
        self, dilations=dilation, kernel_shape=n_dims_one, strides=n_dims_one
    )

    ends = op.Constant(value_ints=n_dims_one)
    starts = op.Constant(value_ints=n_dims_zero)
    axes = op.Constant(value_ints=n_dims_axes)

    delta = op.Slice(flatten_indices, axes=axes, starts=starts, ends=ends)
    indices = op.Sub(indices, delta)

    if self_rank_is_unbatched_rank:
        pool_result = op.Squeeze(pool_result, op.Constant(value_ints=[0]))
        indices = op.Squeeze(indices, op.Constant(value_ints=[0]))

    return (pool_result, indices)


def aten_max_pool3d_with_indices_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
    indices: TensorType,
) -> TensorType:
    """max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor"""

    raise NotImplementedError()


def aten_max_unpool2d(
    self: TensorType, indices: TensorType, output_size: Sequence[int]
) -> TensorType:
    """max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor"""

    raise NotImplementedError()


def aten_max_unpool3d(
    self: TensorType,
    indices: TensorType,
    output_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> TensorType:
    """max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::mish")
def aten_mish(self: TFloat) -> TFloat:
    """mish(Tensor self) -> Tensor"""

    return op.Mish(self)


def aten_mish_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    """mish_backward(Tensor grad_output, Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_mkldnn_linear(
    self: TensorType, weight: TensorType, bias: Optional[TensorType] = None
) -> TensorType:
    """mkldnn_linear(Tensor self, Tensor weight, Tensor? bias=None) -> Tensor"""

    raise NotImplementedError()


def aten_mkldnn_reorder_conv2d_weight(
    self: TensorType,
    padding: Sequence[int] = (0, 0),
    stride: Sequence[int] = (1, 1),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
) -> TensorType:
    """mkldnn_reorder_conv2d_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1) -> Tensor"""

    raise NotImplementedError()


def aten_mkldnn_reorder_conv3d_weight(
    self: TensorType,
    padding: Sequence[int] = (0, 0, 0),
    stride: Sequence[int] = (1, 1, 1),
    dilation: Sequence[int] = (1, 1, 1),
    groups: int = 1,
) -> TensorType:
    """mkldnn_reorder_conv3d_weight(Tensor self, int[3] padding=0, int[3] stride=1, int[3] dilation=1, int groups=1) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::mse_loss", traceable=True)
def aten_mse_loss(self: TReal, target: TReal, reduction: int = 1) -> TReal:
    """mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"""
    # FIXME: When reduction=0, the shape(result) will be different than other case
    result = op.Mul(self - target, self - target)
    if reduction == 1:  # mean
        result = op.ReduceMean(result, keepdims=False)
    if reduction == 2:  # sum
        result = op.ReduceSum(result, keepdims=False)

    return result


def aten_mse_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int
) -> TensorType:
    """mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor"""

    raise NotImplementedError()


def aten_multi_margin_loss(
    self: TensorType,
    target: TensorType,
    p: float = 1.0,
    margin: float = 1.0,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
) -> TensorType:
    """multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


def aten_multi_margin_loss_backward(
    grad_output: TensorType,
    self: TensorType,
    target: TensorType,
    p: float,
    margin: float,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
) -> TensorType:
    """multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


def aten_multilabel_margin_loss(
    self: TensorType, target: TensorType, reduction: int = 1
) -> TensorType:
    """multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


def aten_multilabel_margin_loss_backward(
    grad_output: TensorType,
    self: TensorType,
    target: TensorType,
    reduction: int,
    is_target: TensorType,
) -> TensorType:
    """multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor"""

    raise NotImplementedError()


def aten_multilabel_margin_loss_forward(
    self: TensorType, target: TensorType, reduction: int
) -> tuple[TensorType, TensorType]:
    """multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)"""

    raise NotImplementedError()


@torch_op("aten::nll_loss", traceable=True)
def aten_nll_loss(
    self: TFloat,
    target: INT64,
    reduction: int = 1,
    ignore_index: int = -100,
) -> TFloat:
    """nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor"""

    self_rank_is_1 = Rank(self) == 1
    if self_rank_is_1:  # self rank should be at least 2
        self = op.Unsqueeze(self, op.Constant(value_ints=[0]))

    rank_target = Rank(target)
    if rank_target == 0:  # target rank should be at least 1
        target = op.Unsqueeze(target, op.Constant(value_ints=[0]))

    if reduction == 0:
        result = op.NegativeLogLikelihoodLoss(
            self, target, ignore_index=ignore_index, reduction="none"
        )
    elif reduction == 1:
        result = op.NegativeLogLikelihoodLoss(
            self, target, ignore_index=ignore_index, reduction="mean"
        )
    else:  # assert reduction == 2
        result = op.NegativeLogLikelihoodLoss(
            self, target, ignore_index=ignore_index, reduction="sum"
        )

    if self_rank_is_1:
        result = op.Squeeze(result)

    return result


@torch_op("aten::nll_loss", traceable=True)
def aten_nll_loss_weight(
    self: TFloat,
    target: INT64,
    weight: TFloat,
    reduction: int = 1,
    ignore_index: int = -100,
) -> TFloat:
    """nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor"""

    self_rank_is_1 = Rank(self) == 1
    if self_rank_is_1:
        # self rank should be at least 2
        self = op.Unsqueeze(self, op.Constant(value_ints=[0]))

    rank_target = Rank(target)
    if rank_target == 0:  # target rank should be at least 1
        target = op.Unsqueeze(target, op.Constant(value_ints=[0]))

    if reduction == 0:
        result = op.NegativeLogLikelihoodLoss(
            self, target, weight, ignore_index=ignore_index, reduction="none"
        )
    elif reduction == 1:
        result = op.NegativeLogLikelihoodLoss(
            self, target, weight, ignore_index=ignore_index, reduction="mean"
        )
    else:
        result = op.NegativeLogLikelihoodLoss(
            self, target, weight, ignore_index=ignore_index, reduction="sum"
        )

    if self_rank_is_1:
        result = op.Squeeze(result)

    return result


def aten_nll_loss2d(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
    ignore_index: INT64 = -100,
) -> TensorType:
    """nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor"""

    raise NotImplementedError()


def aten_nll_loss2d_backward(
    grad_output: TensorType,
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType],
    reduction: int,
    ignore_index: INT64,
    total_weight: TensorType,
) -> TensorType:
    """nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight) -> Tensor"""

    raise NotImplementedError()


def aten_nll_loss2d_forward(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType],
    reduction: int,
    ignore_index: INT64,
) -> tuple[TensorType, TensorType]:
    """nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)"""

    raise NotImplementedError()


def aten_nll_loss_backward(
    grad_output: TensorType,
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType],
    reduction: int,
    ignore_index: INT64,
    total_weight: TensorType,
) -> TensorType:
    """nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight) -> Tensor"""

    raise NotImplementedError()


def aten_nll_loss_forward(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType],
    reduction: int,
    ignore_index: INT64,
) -> tuple[TensorType, TensorType]:
    """nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)"""

    raise NotImplementedError()


def aten_nll_loss_nd(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
    ignore_index: INT64 = -100,
) -> TensorType:
    """nll_loss_nd(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor"""

    raise NotImplementedError()


def aten_one_hot(self: TensorType, num_classes: int = -1) -> TensorType:
    """one_hot(Tensor self, int num_classes=-1) -> Tensor"""

    raise NotImplementedError()


def aten_pad(
    self: TensorType, pad: INT64, mode: str = "constant", value: Optional[float] = None
) -> TensorType:
    """pad(Tensor self, SymInt[] pad, str mode="constant", float? value=None) -> Tensor"""

    raise NotImplementedError()


def aten_pad_sequence(
    sequences: Sequence[TensorType], batch_first: bool = False, padding_value: float = 0.0
) -> TensorType:
    """pad_sequence(Tensor[] sequences, bool batch_first=False, float padding_value=0.0) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::reflection_pad1d")
def aten_reflection_pad1d(self: TFloat, padding: INT64) -> TFloat:
    """reflection_pad1d(Tensor self, SymInt[2] padding) -> Tensor"""

    # assert len(padding) == 2
    # Input of padding argument should be [x,y], need change to onnx format [0, x, 0, y]
    start = op.Slice(padding, [0], [1], axes=[0])
    end = op.Slice(padding, [1], [2], axes=[0])
    padding_onnx = op.Concat(
        op.Constant(value_ints=[0]), start, op.Constant(value_ints=[0]), end, axis=0
    )
    return op.Pad(self, padding_onnx, mode="reflect")


def aten_reflection_pad1d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    """reflection_pad1d_backward(Tensor grad_output, Tensor self, SymInt[2] padding) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::reflection_pad2d")
def aten_reflection_pad2d(self: TTensor, padding: INT64) -> TTensor:
    """reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor"""
    # Convert torch padding format to onnx padding format
    # Python code is:
    # dim = len(self.shape)
    # paddings = list(padding[:]) + [0] * (dim * 2 - len(padding))
    # paddings = paddings[-2::-2] + paddings[-1::-2]

    neg_1 = op.Constant(value_ints=[-1])
    zero = op.Constant(value_ints=[0])
    # [0] * (rank * 2 - len(padding))
    rank = Rank(self)
    zero_count = op.Reshape(op.Sub(op.Mul(rank, 2), op.Size(padding)), neg_1)
    zeros = op.Expand(zero, zero_count)
    # list(padding[:]) + [0] * (dim * 2 - len(padding))
    torch_paddings = op.Concat(padding, zeros, axis=0)
    # paddings[-2::-2]
    size_d = op.Size(torch_paddings)
    steps = op.Constant(value_ints=[-2])
    starts = steps
    ends = op.Sub(starts, size_d)
    odd_elements = op.Slice(torch_paddings, starts, ends, zero, steps)
    # paddings[-1::-2]
    starts = neg_1
    ends = op.Sub(starts, size_d)
    even_elements = op.Slice(torch_paddings, starts, ends, zero, steps)
    # paddings[-2::-2] + paddings[-1::-2]
    onnx_padding = op.Concat(odd_elements, even_elements, axis=0)

    return op.Pad(self, onnx_padding, mode="reflect")


def aten_reflection_pad2d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    """reflection_pad2d_backward(Tensor grad_output, Tensor self, SymInt[4] padding) -> Tensor"""

    raise NotImplementedError()


def aten_reflection_pad3d(self: TensorType, padding: INT64) -> TensorType:
    """reflection_pad3d(Tensor self, SymInt[6] padding) -> Tensor"""

    raise NotImplementedError()


def aten_reflection_pad3d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    """reflection_pad3d_backward(Tensor grad_output, Tensor self, SymInt[6] padding) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::relu")
def aten_relu(self: TReal) -> TReal:
    """relu(Tensor self) -> Tensor"""

    return op.Relu(self)


@torch_op("aten::relu6", traceable=True)
def aten_relu6(self: TReal) -> TReal:
    """relu6(Tensor self) -> Tensor"""

    six = op.CastLike(op.Constant(value_int=6), self)
    return op.Min(op.Relu(self), six)


@torch_op("aten::replication_pad1d")
def aten_replication_pad1d(self: TensorType, padding: INT64) -> TensorType:
    """replication_pad1d(Tensor self, SymInt[2] padding) -> Tensor"""

    # assert len(padding) == 2
    # Input of padding argument should be [x,y], need change to onnx format [0, x, 0, y]
    start = op.Slice(padding, [0], [1], axes=[0])
    end = op.Slice(padding, [1], [2], axes=[0])
    padding_onnx = op.Concat(
        op.Constant(value_ints=[0]), start, op.Constant(value_ints=[0]), end, axis=0
    )
    return op.Pad(self, padding_onnx, mode="edge")


def aten_replication_pad1d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    """replication_pad1d_backward(Tensor grad_output, Tensor self, SymInt[2] padding) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::replication_pad2d")
def aten_replication_pad2d(self: TTensor, padding: INT64) -> TTensor:
    """replication_pad2d(Tensor self, SymInt[4] padding) -> Tensor"""

    neg_1 = op.Constant(value_ints=[-1])
    zero = op.Constant(value_ints=[0])
    # [0] * (rank * 2 - len(padding))
    rank = Rank(self)
    zero_count = op.Reshape(op.Sub(op.Mul(rank, 2), op.Size(padding)), neg_1)
    zeros = op.Expand(zero, zero_count)
    # list(padding[:]) + [0] * (dim * 2 - len(padding))
    torch_paddings = op.Concat(padding, zeros, axis=0)
    # paddings[-2::-2]
    size_d = op.Size(torch_paddings)
    steps = op.Constant(value_ints=[-2])
    starts = steps
    ends = op.Sub(starts, size_d)
    odd_elements = op.Slice(torch_paddings, starts, ends, zero, steps)
    # paddings[-1::-2]
    starts = neg_1
    ends = op.Sub(starts, size_d)
    even_elements = op.Slice(torch_paddings, starts, ends, zero, steps)
    # paddings[-2::-2] + paddings[-1::-2]
    onnx_padding = op.Concat(odd_elements, even_elements, axis=0)

    return op.Pad(self, onnx_padding, mode="edge")


def aten_replication_pad2d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    """replication_pad2d_backward(Tensor grad_output, Tensor self, SymInt[4] padding) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::replication_pad3d")
def aten_replication_pad3d(self: TTensor, padding: INT64) -> TTensor:
    """replication_pad3d(Tensor self, SymInt[6] padding) -> Tensor"""

    neg_1 = op.Constant(value_ints=[-1])
    zero = op.Constant(value_ints=[0])
    # [0] * (rank * 2 - len(padding))
    rank = Rank(self)
    zero_count = op.Reshape(op.Sub(op.Mul(rank, 2), op.Size(padding)), neg_1)
    zeros = op.Expand(zero, zero_count)
    # list(padding[:]) + [0] * (dim * 2 - len(padding))
    torch_paddings = op.Concat(padding, zeros, axis=0)
    # paddings[-2::-2]
    size_d = op.Size(torch_paddings)
    steps = op.Constant(value_ints=[-2])
    starts = steps
    ends = op.Sub(starts, size_d)
    odd_elements = op.Slice(torch_paddings, starts, ends, zero, steps)
    # paddings[-1::-2]
    starts = neg_1
    ends = op.Sub(starts, size_d)
    even_elements = op.Slice(torch_paddings, starts, ends, zero, steps)
    # paddings[-2::-2] + paddings[-1::-2]
    onnx_padding = op.Concat(odd_elements, even_elements, axis=0)

    return op.Pad(self, onnx_padding, mode="edge")


def aten_replication_pad3d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    """replication_pad3d_backward(Tensor grad_output, Tensor self, SymInt[6] padding) -> Tensor"""

    raise NotImplementedError()


def aten_rrelu_with_noise(
    self: TensorType,
    noise: TensorType,
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
    training: bool = False,
    generator: Optional[str] = None,
) -> TensorType:
    """rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor"""

    raise NotImplementedError()


def aten_rrelu_with_noise_backward(
    grad_output: TensorType,
    self: TensorType,
    noise: TensorType,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool,
) -> TensorType:
    """rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::scaled_dot_product_attention", private=True)
def _causal_attention_mask(query: TFloat, key: TFloat) -> TFloat:
    """Create a causal mask for the given query and key tensors.

    Equivalent to::
        mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_mask = torch.zeros(L, S, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(not mask, -float('inf'))

    Args:
        query: Tensor of shape [..., L, E]
        key: Tensor of shape [..., S, E]

    Returns:
        Tensor of shape [L, S]
    """
    target_length = op.Shape(query)[-2:-1]
    source_length = op.Shape(key)[-2:-1]
    # attn_mask = torch.ones(L, S) := {
    size = op.Concat(target_length, source_length, axis=0)
    attn_mask = op.Expand(1.0, size)
    # }
    attn_mask = op.Trilu(attn_mask, upper=0)
    # The causal mask has 0s in the lower triangle and -inf in the upper triangle.
    attn_mask = op.Where(op.Equal(attn_mask, 0.0), op.Constant(value_float=-float("inf")), 0.0)
    attn_mask = op.CastLike(attn_mask, query)
    return attn_mask


@torch_op("aten::scaled_dot_product_attention", private=True)
def _attention_scale(query: TFloat) -> TFloat:
    """Calculate the scale factor for the attention result.

    Args:
        query: Tensor of shape [..., L, E]

    Returns:
        Scalar scale factor := 1 / math.sqrt(query.size(-1))
    """
    embedding_size = op.CastLike(op.Shape(query)[-1], query)
    scale = op.Div(1.0, op.Sqrt(embedding_size))
    return scale


@torch_op("aten::scaled_dot_product_attention", trace_only=True)
def aten_scaled_dot_product_attention(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    attn_mask: Optional[TFloat] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> TFloat:
    """scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None) -> Tensor

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    Equivalent to the PyTorch code::
        scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
        attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
        attn_weight = torch.softmax((Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ V

    where Q, K, V are the query, key, and value tensors, respectively.
    L is the target sequence length, S is the source sequence length, and E is the embedding size.
    """
    # Use trace_only to handle optional inputs
    assert (not is_causal) or (
        is_causal and attn_mask is None
    ), "is_causal and attn_mask cannot be set at the same time"

    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    if scale is None:
        scale = _attention_scale(query)
    scale = op.CastLike(scale, query)

    if is_causal:
        attn_mask = _causal_attention_mask(query, key)

    if attn_mask is None:
        return _aten_scaled_dot_product_attention_no_mask_onnx(
            query, key, value, scale, dropout_p
        )

    return _aten_scaled_dot_product_attention_float_mask_onnx(
        query, key, value, attn_mask, scale, dropout_p
    )


@torch_op("aten::_scaled_dot_product_flash_attention", private=True)
def _aten__scaled_dot_product_flash_attention_fillin_empty_outputs(
    query: TFloat,
) -> Tuple[FLOAT, INT64, INT64, FLOAT]:
    query_first_three_dims = op.Slice(
        op.Shape(query), op.Constant(value_ints=[0]), op.Constant(value_ints=[3])
    )
    logsumexp = op.Expand(0.0, query_first_three_dims)
    # TODO: Eliminate `make_tensor` usage when ORT supports empty tensor.
    empty_tensor_int = op.Cast(
        op.ConstantOfShape(
            op.Constant(value=onnx.helper.make_tensor("Empty_INTS", INT64.dtype, [0], []))
        ),
        to=INT64.dtype,
    )
    empty_tensor_float = op.ConstantOfShape(
        op.Constant(value=onnx.helper.make_tensor("Empty_FLOATS", INT64.dtype, [0], []))
    )
    empty_int = op.Constant(value_int=0)

    return logsumexp, empty_tensor_int, empty_int, empty_tensor_float


@torch_op("aten::_scaled_dot_product_flash_attention", trace_only=True)
def aten__scaled_dot_product_flash_attention(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: Optional[float] = None,
) -> Tuple[TFloat, FLOAT, INT64, INT64, INT64, INT64, INT64, INT64, FLOAT]:
    """_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0.0, bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, int max_q, int max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)

    One of the implementations of scaled_dot_product_attention.
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    NOTE: Currently, there are three implementations of nn.scaled_dot_product_attention in PyTorch due to optimization.
    However, it's the same implementation from ONNX perspective.

    """
    result = aten_scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )

    # The followings are not comsumed by the graph.
    (
        logsumexp,
        empty_tensor_int,
        empty_int,
        empty_tensor_float,
    ) = _aten__scaled_dot_product_flash_attention_fillin_empty_outputs(query)

    return (
        result,
        logsumexp,
        empty_tensor_int,
        empty_tensor_int,
        empty_int,
        empty_int,
        empty_tensor_int,
        empty_tensor_int,
        empty_tensor_float,
    )


@torch_op("aten::_scaled_dot_product_efficient_attention", private=True)
def _aten_scaled_dot_product_efficient_attention_fillin_empty_outputs(
    query: TFloat,
    compute_log_sumexp: bool,
) -> Tuple[FLOAT, INT64]:
    """_scaled_dot_product_efficient_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0.0, bool is_causal=False, *, float? scale=None) -> (Tensor output, Tensor log_sumexp, Tensor philox_seed, Tensor philox_offset)"""

    query = op.Transpose(query, perm=[0, 2, 1, 3])
    query_shape = op.Shape(query)
    query_first_dims = query_shape[:1]
    query_second_dims = query_shape[1:2]
    num_heads = query_shape[-2:-1]

    if compute_log_sumexp:
        logsumexp_dim = op.Cast(
            op.Ceil(op.Cast(query_second_dims, to=FLOAT.dtype) / 32.0) * 32.0, to=INT64.dtype
        )
        logsum_exp = op.Expand(
            0.0, op.Concat(query_first_dims, num_heads, logsumexp_dim, axis=0)
        )
    else:
        logsum_exp = op.Expand(0.0, op.Concat(query_first_dims, num_heads, [0], axis=0))

    # See Note [Seed and Offset]:
    empty_tensor_int = op.Cast(
        op.ConstantOfShape(
            op.Constant(value=onnx.helper.make_tensor("Empty_INTS", INT64.dtype, [0], []))
        ),
        to=INT64.dtype,
    )

    return logsum_exp, empty_tensor_int


@torch_op("aten::_scaled_dot_product_flash_attention_for_cpu", trace_only=True)
def aten__scaled_dot_product_flash_attention_for_cpu(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    attn_mask: Optional[TFloat] = None,
    scale: Optional[float] = None,
) -> Tuple[TFloat, FLOAT]:
    """_scaled_dot_product_flash_attention_for_cpu(Tensor query, Tensor key, Tensor value, float dropout_p=0.0, bool is_causal=False, *, Tensor? attn_mask=None, float? scale=None) -> (Tensor output, Tensor logsumexp)"""
    result = aten_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    query_shape = op.Shape(query)
    query_first_dims = op.Slice(query_shape, [0], [1])
    query_second_dims = op.Slice(query_shape, [1], [2])
    num_heads = op.Slice(query_shape, [-2], [-1])
    logsumexp_dim = op.Cast(
        op.Ceil(op.Cast(query_second_dims, to=FLOAT.dtype) / 32.0) * 32.0, to=INT64.dtype
    )
    logsum_exp = op.Expand(0.0, op.Concat(query_first_dims, num_heads, logsumexp_dim, axis=0))
    return result, logsum_exp


@torch_op("aten::_scaled_dot_product_efficient_attention", trace_only=True)
def aten__scaled_dot_product_efficient_attention(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    attn_bias: Optional[TFloat],
    compute_log_sumexp: bool,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[TFloat, FLOAT, INT64, INT64]:
    """_scaled_dot_product_efficient_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0.0, bool is_causal=False, *, float? scale=None) -> (Tensor output, Tensor log_sumexp, Tensor philox_seed, Tensor philox_offset)"""

    result = aten_scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )

    # The followings are not comsumed by the graph.
    (
        logsumexp,
        empty_tensor_int,
    ) = _aten_scaled_dot_product_efficient_attention_fillin_empty_outputs(
        query, compute_log_sumexp
    )

    return (
        result,
        logsumexp,
        empty_tensor_int,
        empty_tensor_int,
    )


@torch_op("aten::scaled_dot_product_attention", trace_only=True)
def aten_scaled_dot_product_attention_bool_mask(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    attn_mask: Optional[BOOL] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> TFloat:
    """scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None) -> Tensor

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    Equivalent to the PyTorch code::
        scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
        attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
        attn_weight = torch.softmax((Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ V

    where Q, K, V are the query, key, and value tensors, respectively.
    L is the target sequence length, S is the source sequence length, and E is the embedding size.
    """
    # Use trace_only to handle optional inputs
    assert (not is_causal) or (
        is_causal and attn_mask is None
    ), "is_causal and attn_mask cannot be set at the same time"

    if scale is None:
        scale = _attention_scale(query)
    scale = op.CastLike(scale, query)

    if is_causal:
        attn_mask = _causal_attention_mask(query, key)
        # The causal mask is always float
        return _aten_scaled_dot_product_attention_float_mask_onnx(
            query, key, value, attn_mask, scale, dropout_p
        )

    if attn_mask is None:
        return _aten_scaled_dot_product_attention_no_mask_onnx(
            query, key, value, scale, dropout_p
        )

    return _aten_scaled_dot_product_attention_bool_mask_onnx(
        query, key, value, attn_mask, scale, dropout_p
    )


@torch_op("aten::scaled_dot_product_attention", private=True)
def _aten_scaled_dot_product_attention_no_mask_onnx(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    scale: TFloat,
    dropout_p: float,
) -> TFloat:
    # Swap the last two axes of key
    key_shape = op.Shape(key)
    key_last_dim = key_shape[-1:]
    key_second_last_dim = key_shape[-2:-1]
    key_first_dims = key_shape[:-2]
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(key_first_dims, key_last_dim, key_second_last_dim, axis=0)
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(key_transposed, op.CastLike(op.Sqrt(scale), key_transposed))
    attn_weight = op.Softmax(
        op.MatMul(query_scaled, key_transposed_scaled),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p)
    return op.MatMul(attn_weight, value)


@torch_op("aten::scaled_dot_product_attention", private=True)
def _aten_scaled_dot_product_attention_bool_mask_onnx(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    attn_mask: BOOL,
    scale: TFloat,
    dropout_p: float,
) -> TFloat:
    # Swap the last two axes of key
    key_shape = op.Shape(key)
    key_last_dim = key_shape[-1:]
    key_second_last_dim = key_shape[-2:-1]
    key_first_dims = key_shape[:-2]
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(key_first_dims, key_last_dim, key_second_last_dim, axis=0)
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(key_transposed, op.Sqrt(scale))
    # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf'))
    attn_mask = op.Where(attn_mask, 0.0, op.Constant(value_float=-float("inf")))
    attn_weight = op.Softmax(
        op.Add(op.MatMul(query_scaled, key_transposed_scaled), attn_mask),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p)
    return op.MatMul(attn_weight, value)


@torch_op("aten::scaled_dot_product_attention", private=True)
def _aten_scaled_dot_product_attention_float_mask_onnx(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    attn_mask: TFloat,
    scale: TFloat,
    dropout_p: float,
) -> TFloat:
    # Swap the last two axes of key
    key_shape = op.Shape(key)
    key_last_dim = key_shape[-1:]
    key_second_last_dim = key_shape[-2:-1]
    key_first_dims = key_shape[:-2]
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(key_first_dims, key_last_dim, key_second_last_dim, axis=0)
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(key_transposed, op.Sqrt(scale))
    attn_weight = op.Softmax(
        op.Add(op.MatMul(query_scaled, key_transposed_scaled), attn_mask),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p)
    return op.MatMul(attn_weight, value)


def aten_sigmoid_backward(grad_output: TensorType, output: TensorType) -> TensorType:
    """sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::silu", traceable=True)
def aten_silu(self: TFloat) -> TFloat:
    """silu(Tensor self) -> Tensor"""

    return op.Mul(self, op.Sigmoid(self))


def aten_silu_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    """silu_backward(Tensor grad_output, Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_slow_conv3d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1, 1, 1),
    padding: INT64 = (0, 0, 0),
) -> TensorType:
    """slow_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, SymInt[3] padding=0) -> Tensor"""

    raise NotImplementedError()


def aten_slow_conv3d_forward(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: INT64,
) -> TensorType:
    """slow_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, SymInt[3] padding) -> Tensor"""

    raise NotImplementedError()


def aten_slow_conv_dilated2d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1, 1),
    padding: INT64 = (0, 0),
    dilation: Sequence[int] = (1, 1),
) -> TensorType:
    """slow_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, SymInt[2] padding=0, int[2] dilation=1) -> Tensor"""

    raise NotImplementedError()


def aten_slow_conv_dilated3d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1, 1, 1),
    padding: INT64 = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
) -> TensorType:
    """slow_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, SymInt[3] padding=0, int[3] dilation=1) -> Tensor"""

    raise NotImplementedError()


def aten_slow_conv_transpose2d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1, 1),
    padding: INT64 = (0, 0),
    output_padding: INT64 = (0, 0),
    dilation: Sequence[int] = (1, 1),
) -> TensorType:
    """slow_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, SymInt[2] padding=0, SymInt[2] output_padding=0, int[2] dilation=1) -> Tensor"""

    raise NotImplementedError()


def aten_slow_conv_transpose3d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1, 1, 1),
    padding: INT64 = (0, 0, 0),
    output_padding: INT64 = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
) -> TensorType:
    """slow_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, SymInt[3] padding=0, SymInt[3] output_padding=0, int[3] dilation=1) -> Tensor"""

    raise NotImplementedError()


def aten_smooth_l1_loss(
    self: TensorType, target: TensorType, reduction: int = 1, beta: float = 1.0
) -> TensorType:
    """smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean, float beta=1.0) -> Tensor"""

    raise NotImplementedError()


def aten_smooth_l1_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int, beta: float
) -> TensorType:
    """smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta) -> Tensor"""

    raise NotImplementedError()


def aten_soft_margin_loss(
    self: TensorType, target: TensorType, reduction: int = 1
) -> TensorType:
    """soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


def aten_soft_margin_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int
) -> TensorType:
    """soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::softplus")
def aten_softplus(self: TFloat, beta: float = 1.0, threshold: float = 20.0) -> TFloat:
    """softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor"""

    self_scaled = self * beta
    softplus = op.Softplus(self_scaled) / beta
    return op.Where(self_scaled > threshold, self, softplus)


def aten_softplus_backward(
    grad_output: TensorType, self: TensorType, beta: float, threshold: float
) -> TensorType:
    """softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold) -> Tensor"""

    raise NotImplementedError()


def aten_softshrink(self: TensorType, lambd: float = 0.5) -> TensorType:
    """softshrink(Tensor self, Scalar lambd=0.5) -> Tensor"""

    raise NotImplementedError()


def aten_softshrink_backward(
    grad_output: TensorType, self: TensorType, lambd: float
) -> TensorType:
    """softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor"""

    raise NotImplementedError()


def aten_tanh_backward(grad_output: TensorType, output: TensorType) -> TensorType:
    """tanh_backward(Tensor grad_output, Tensor output) -> Tensor"""

    raise NotImplementedError()


def aten_thnn_conv2d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
) -> TensorType:
    """thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0) -> Tensor"""

    raise NotImplementedError()


def aten_unflatten_dense_tensors(
    flat: TensorType, tensors: Sequence[TensorType]
) -> TensorType:
    """unflatten_dense_tensors(Tensor flat, Tensor[] tensors) -> Tensor[]"""

    raise NotImplementedError()


def _get_upsample_align_corners_mode(align_corners: bool) -> str:
    return "align_corners" if align_corners else "pytorch_half_pixel"


@torch_op(
    (
        "aten::upsample_bicubic2d",
        "aten::upsample_bilinear2d",
        "aten::upsample_nearest1d",
        "aten::upsample_nearest2d",
        "aten::upsample_nearest3d",
        "aten::upsample_trilinear3d",
    ),
    private=True,
)
def _aten_upsample_output_size(
    self: TReal,
    output_size: INT64,
    mode: str,
    coordinate_transformation_mode: str,
) -> TReal:
    self_shape = op.Shape(self)
    starts = op.Constant(value_ints=[0])
    ends = op.Constant(value_ints=[2])
    batch_channel = op.Slice(self_shape, starts, ends)
    output_size = op.Concat(batch_channel, output_size, axis=0)
    return op.Resize(
        self,
        None,
        None,
        output_size,
        mode=mode,
        coordinate_transformation_mode=coordinate_transformation_mode,
        nearest_mode="floor",
    )


@torch_op(("aten::upsample_bicubic2d", "aten::upsample_bilinear2d"), private=True)
def _aten_upsample_scales(
    self: TReal,
    scale_factors: TFloat,
    mode: str,
    coordinate_transformation_mode: str,
) -> TReal:
    scale_factors = op.Cast(scale_factors, to=FLOAT.dtype)
    scale_factors = op.Concat(op.Constant(value_floats=[1.0, 1.0]), scale_factors, axis=0)
    return op.Resize(
        self,
        None,
        scale_factors,  # format should be: [1.0, 1.0, scale_h, scale_w]
        None,
        mode=mode,
        coordinate_transformation_mode=coordinate_transformation_mode,
    )


@torch_op("aten::upsample_bicubic2d", trace_only=True)
def aten_upsample_bicubic2d(
    self: TReal,
    output_size: INT64,
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TReal:
    """upsample_bicubic2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor"""

    # NOTE: Based on experimentation, scales_h and scales_w are always ignored in PyTorch,
    # unless when align_corners is True, in which case we do not know what is going on.
    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    return _aten_upsample_output_size(
        self,
        output_size,
        mode="cubic",
        coordinate_transformation_mode=coordinate_transformation_mode,
    )


@torch_op("aten::upsample_bicubic2d.vec", trace_only=True)
def aten_upsample_bicubic2d_vec(
    self: TReal,
    output_size: INT64,
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> TReal:
    """upsample_bicubic2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor"""

    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    if scale_factors is not None:
        result = _aten_upsample_scales(
            self,
            op.Constant(value_floats=scale_factors),
            mode="cubic",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )
    else:
        result = _aten_upsample_output_size(
            self,
            output_size,
            mode="cubic",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )

    return result


def aten_upsample_bicubic2d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    """upsample_bicubic2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::upsample_bilinear2d", trace_only=True)
def aten_upsample_bilinear2d(
    self: TReal,
    output_size: INT64,
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TReal:
    """upsample_bilinear2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor"""

    # NOTE: Based on experimentation, scales_h and scales_w are always ignored in PyTorch,
    # unless when align_corners is True, in which case we do not know what is going on.
    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    return _aten_upsample_output_size(
        self,
        output_size,
        coordinate_transformation_mode=coordinate_transformation_mode,
        mode="linear",
    )


@torch_op("aten::upsample_bilinear2d.vec", trace_only=True)
def aten_upsample_bilinear2d_vec(
    self: TReal,
    output_size: Optional[INT64],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> TReal:
    """upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor"""

    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    if scale_factors is not None:
        result = _aten_upsample_scales(
            self,
            op.Constant(value_floats=scale_factors),
            mode="linear",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )
    else:
        result = _aten_upsample_output_size(
            self,
            output_size,
            mode="linear",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )

    return result


def aten_upsample_bilinear2d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    """upsample_bilinear2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::upsample_linear1d", trace_only=True)
def aten_upsample_linear1d(
    self: TReal, output_size: INT64, align_corners: bool, scales: Optional[float] = None
) -> TReal:
    """upsample_linear1d(Tensor self, SymInt[1] output_size, bool align_corners, float? scales=None) -> Tensor"""
    # FIXME(justinchuby): Support when scales is provided and align_corners is False
    del scales
    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    return _aten_upsample_output_size(
        self,
        output_size,
        mode="linear",
        coordinate_transformation_mode=coordinate_transformation_mode,
    )


def aten_upsample_linear1d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    align_corners: bool,
    scales: Optional[float] = None,
) -> TensorType:
    """upsample_linear1d_backward(Tensor grad_output, SymInt[1] output_size, SymInt[3] input_size, bool align_corners, float? scales=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::upsample_nearest1d", trace_only=True)
def aten_upsample_nearest1d(
    self: TReal, size: INT64, scale_factor: Optional[float] = None
) -> TReal:
    """upsample_nearest1d(Tensor self, SymInt[1] output_size, float? scales=None) -> Tensor"""
    if size is not None:
        return _aten_upsample_output_size(self, size, "nearest", "asymmetric")
    else:
        return _aten_upsample_nearest1d_scales(self, scale_factor)


@torch_op("aten::upsample_nearest1d", private=True)
def _aten_upsample_nearest1d_scales(
    self: TReal,
    scale_factors: TFloat,
) -> TReal:
    scale_factors = op.Cast(scale_factors, to=FLOAT.dtype)
    scale_factors = op.Concat(op.Constant(value_floats=[1.0, 1.0]), scale_factors, axis=0)
    return op.Resize(
        self,
        None,
        scale_factors,  # format should be: [1.0, 1.0, scale_h, scale_w]
        None,
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )


def aten_upsample_nearest1d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    scales: Optional[float] = None,
) -> TensorType:
    """upsample_nearest1d_backward(Tensor grad_output, SymInt[1] output_size, SymInt[3] input_size, float? scales=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::upsample_nearest2d", trace_only=True)
def aten_upsample_nearest2d(
    self: TReal,
    size: INT64,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TReal:
    """upsample_nearest2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor"""

    # NOTE: trace_only because optional attributes are not supported by ONNX
    # TODO(justinchuby): Conditionally use scales
    del scales_h
    del scales_w

    return _aten_upsample_output_size(self, size, "nearest", "asymmetric")


def aten_upsample_nearest2d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    """upsample_nearest2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::upsample_nearest3d", trace_only=True)
def aten_upsample_nearest3d(
    self: TReal,
    size: INT64,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TReal:
    """upsample_nearest3d(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor"""

    del scales_h
    del scales_w
    del scales_d

    return _aten_upsample_output_size(self, size, "nearest", "asymmetric")


def aten_upsample_nearest3d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    """upsample_nearest3d_backward(Tensor grad_output, SymInt[3] output_size, SymInt[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::upsample_trilinear3d", trace_only=True)
def aten_upsample_trilinear3d(
    self: TReal,
    output_size: INT64,
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TReal:
    """upsample_trilinear3d(Tensor self, SymInt[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor"""

    del scales_d
    del scales_h
    del scales_w

    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    return _aten_upsample_output_size(
        self,
        output_size,
        mode="linear",
        coordinate_transformation_mode=coordinate_transformation_mode,
    )


@torch_op("aten::upsample_trilinear3d.vec", trace_only=True)
def aten_upsample_trilinear3d_vec(
    self: TReal,
    output_size: INT64,
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> TReal:
    """upsample_trilinear3d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor"""

    coordinate_transformation_mode = _get_upsample_align_corners_mode(align_corners)
    if scale_factors is not None:
        result = _aten_upsample_scales(
            self,
            op.Constant(value_floats=scale_factors),
            mode="linear",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )
    else:
        result = _aten_upsample_output_size(
            self,
            output_size,
            mode="linear",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )
    return result


def aten_upsample_trilinear3d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    """upsample_trilinear3d_backward(Tensor grad_output, SymInt[3] output_size, SymInt[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor"""

    raise NotImplementedError()
