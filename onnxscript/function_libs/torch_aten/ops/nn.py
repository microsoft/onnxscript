# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""torch.ops.aten operators under the `nn` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""
from __future__ import annotations

from typing import Optional, Sequence

from beartype.vale import Is
from typing_extensions import Annotated

from onnxscript import INT64, TensorType
from onnxscript.function_libs.torch_aten.typing import FloatType
from onnxscript.onnx_opset import opset18 as op


def aten_adaptive_avg_pool2d(self: TensorType, output_size: INT64[2]) -> TensorType:
    raise NotImplementedError()


def aten_adaptive_avg_pool3d(self: TensorType, output_size: INT64[3]) -> TensorType:
    raise NotImplementedError()


def aten_adaptive_max_pool2d(
    self: TensorType, output_size: Sequence[int]
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_adaptive_max_pool2d_backward(
    grad_output: TensorType, self: TensorType, indices: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_adaptive_max_pool3d(
    self: TensorType, output_size: Sequence[int]
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_adaptive_max_pool3d_backward(
    grad_output: TensorType, self: TensorType, indices: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_avg_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> TensorType:
    raise NotImplementedError()


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
    raise NotImplementedError()


def aten_avg_pool3d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> TensorType:
    raise NotImplementedError()


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
    raise NotImplementedError()


def aten_binary_cross_entropy(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = "mean",
) -> TensorType:
    raise NotImplementedError()


def aten_binary_cross_entropy_backward(
    grad_output: TensorType,
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = "mean",
) -> TensorType:
    raise NotImplementedError()


def aten_col2im(
    self: TensorType,
    output_size: INT64[2],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
) -> TensorType:
    raise NotImplementedError()


def aten_conv_depthwise3d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: INT64[3],
    dilation: Sequence[int],
) -> TensorType:
    raise NotImplementedError()


def aten_cross_entropy_loss(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = "mean",
    ignore_index: INT64 = -100,
    label_smoothing: float = 0.0,
) -> TensorType:
    raise NotImplementedError()


def aten_elu(
    self: FloatType,
    alpha: float = 1.0,
    scale: Annotated[float, Is[lambda x: x == 1.0]] = 1.0,
    input_scale: Annotated[float, Is[lambda x: x == 1.0]] = 1.0,
) -> TensorType:
    # del scale
    # del input_scale
    return op.Elu(self, alpha=alpha)


def aten_elu__int(
    self: IntType,
    alpha: float = 1.0,
    scale: Annotated[float, Is[lambda x: x == 1.0]] = 1.0,
    input_scale: Annotated[float, Is[lambda x: x == 1.0]] = 1.0,
) -> TensorType:
    # TODO(justinchuby): Move the type casting logic to exporter?
    # del scale
    # del input_scale
    return op.Elu(op.Cast(self, to=onnxscript.FLOAT), alpha=alpha)


def aten_elu_backward(
    grad_output: TensorType,
    alpha: float,
    scale: float,
    input_scale: float,
    is_result: bool,
    self_or_result: TensorType,
) -> TensorType:
    raise NotImplementedError()


def aten_flatten_dense_tensors(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_fractional_max_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    random_samples: TensorType,
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_fractional_max_pool2d_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    indices: TensorType,
) -> TensorType:
    raise NotImplementedError()


def aten_fractional_max_pool3d(
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    random_samples: TensorType,
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_fractional_max_pool3d_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    indices: TensorType,
) -> TensorType:
    raise NotImplementedError()


def aten_gelu(self: TensorType, *, approximate: str = "none") -> TensorType:
    raise NotImplementedError()


def aten_gelu_backward(
    grad_output: TensorType, self: TensorType, *, approximate: str = "none"
) -> TensorType:
    raise NotImplementedError()


def aten_glu(self: TensorType, dim: int = -1) -> TensorType:
    raise NotImplementedError()


def aten_glu_backward(grad_output: TensorType, self: TensorType, dim: int) -> TensorType:
    raise NotImplementedError()


def aten_glu_backward_jvp(
    grad_x: TensorType,
    grad_glu: TensorType,
    x: TensorType,
    dgrad_glu: TensorType,
    dx: TensorType,
    dim: int,
) -> TensorType:
    raise NotImplementedError()


def aten_glu_jvp(glu: TensorType, x: TensorType, dx: TensorType, dim: int) -> TensorType:
    raise NotImplementedError()


def aten_hardsigmoid(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_hardsigmoid_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_hardswish(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_hardswish_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_hardtanh(self: TensorType, min_val: int = -1, max_val: int = 1) -> TensorType:
    raise NotImplementedError()


def aten_hardtanh_backward(
    grad_output: TensorType, self: TensorType, min_val: float, max_val: float
) -> TensorType:
    raise NotImplementedError()


def aten_huber_loss(
    self: TensorType, target: TensorType, reduction: int = "mean", delta: float = 1.0
) -> TensorType:
    raise NotImplementedError()


def aten_huber_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int, delta: float
) -> TensorType:
    raise NotImplementedError()


def aten_im2col(
    self: TensorType,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
) -> TensorType:
    raise NotImplementedError()


def aten_infinitely_differentiable_gelu_backward(
    grad: TensorType, self: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_l1_loss(self: TensorType, target: TensorType, reduction: int = "mean") -> TensorType:
    raise NotImplementedError()


def aten_leaky_relu(self: TensorType, negative_slope: float = 0.01) -> TensorType:
    raise NotImplementedError()


def aten_leaky_relu_backward(
    grad_output: TensorType, self: TensorType, negative_slope: float, self_is_result: bool
) -> TensorType:
    raise NotImplementedError()


def aten_linear(
    input: TensorType, weight: TensorType, bias: Optional[TensorType] = None
) -> TensorType:
    raise NotImplementedError()


def aten_log_sigmoid(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_log_sigmoid_backward(
    grad_output: TensorType, self: TensorType, buffer: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_log_sigmoid_forward(self: TensorType) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_logit_backward(
    grad_output: TensorType, self: TensorType, eps: Optional[float] = None
) -> TensorType:
    raise NotImplementedError()


def aten_max_pool2d_with_indices(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


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
    raise NotImplementedError()


def aten_max_pool3d_with_indices(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


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
    raise NotImplementedError()


def aten_max_unpool2d(
    self: TensorType, indices: TensorType, output_size: Sequence[int]
) -> TensorType:
    raise NotImplementedError()


def aten_max_unpool3d(
    self: TensorType,
    indices: TensorType,
    output_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> TensorType:
    raise NotImplementedError()


def aten_mish(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_mish_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_mkldnn_linear(
    self: TensorType, weight: TensorType, bias: Optional[TensorType] = None
) -> TensorType:
    raise NotImplementedError()


def aten_mkldnn_reorder_conv2d_weight(
    self: TensorType,
    padding: Sequence[int] = 0,
    stride: Sequence[int] = 1,
    dilation: Sequence[int] = 1,
    groups: int = 1,
) -> TensorType:
    raise NotImplementedError()


def aten_mkldnn_reorder_conv3d_weight(
    self: TensorType,
    padding: Sequence[int] = 0,
    stride: Sequence[int] = 1,
    dilation: Sequence[int] = 1,
    groups: int = 1,
) -> TensorType:
    raise NotImplementedError()


def aten_mse_loss(self: TensorType, target: TensorType, reduction: int = "mean") -> TensorType:
    raise NotImplementedError()


def aten_mse_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int
) -> TensorType:
    raise NotImplementedError()


def aten_multi_margin_loss(
    self: TensorType,
    target: TensorType,
    p: int = 1,
    margin: int = 1,
    weight: Optional[TensorType] = None,
    reduction: int = "mean",
) -> TensorType:
    raise NotImplementedError()


def aten_multi_margin_loss_backward(
    grad_output: TensorType,
    self: TensorType,
    target: TensorType,
    p: float,
    margin: float,
    weight: Optional[TensorType] = None,
    reduction: int = "mean",
) -> TensorType:
    raise NotImplementedError()


def aten_multilabel_margin_loss(
    self: TensorType, target: TensorType, reduction: int = "mean"
) -> TensorType:
    raise NotImplementedError()


def aten_multilabel_margin_loss_backward(
    grad_output: TensorType,
    self: TensorType,
    target: TensorType,
    reduction: int,
    is_target: TensorType,
) -> TensorType:
    raise NotImplementedError()


def aten_multilabel_margin_loss_forward(
    self: TensorType, target: TensorType, reduction: int
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_nll_loss(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = "mean",
    ignore_index: INT64 = -100,
) -> TensorType:
    raise NotImplementedError()


def aten_nll_loss2d(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = "mean",
    ignore_index: INT64 = -100,
) -> TensorType:
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
    raise NotImplementedError()


def aten_nll_loss2d_forward(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType],
    reduction: int,
    ignore_index: INT64,
) -> tuple[TensorType, TensorType]:
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
    raise NotImplementedError()


def aten_nll_loss_forward(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType],
    reduction: int,
    ignore_index: INT64,
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_nll_loss_nd(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = "mean",
    ignore_index: INT64 = -100,
) -> TensorType:
    raise NotImplementedError()


def aten_one_hot(self: TensorType, num_classes: int = -1) -> TensorType:
    raise NotImplementedError()


def aten_pad(
    self: TensorType, pad: INT64[...], mode: str = "constant", value: Optional[float] = None
) -> TensorType:
    raise NotImplementedError()


def aten_pad_sequence(
    sequences: TensorType[...], batch_first: bool = False, padding_value: float = 0.0
) -> TensorType:
    raise NotImplementedError()


def aten_reflection_pad1d(self: TensorType, padding: INT64[2]) -> TensorType:
    raise NotImplementedError()


def aten_reflection_pad1d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64[2]
) -> TensorType:
    raise NotImplementedError()


def aten_reflection_pad2d(self: TensorType, padding: INT64[4]) -> TensorType:
    raise NotImplementedError()


def aten_reflection_pad2d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64[4]
) -> TensorType:
    raise NotImplementedError()


def aten_reflection_pad3d(self: TensorType, padding: INT64[6]) -> TensorType:
    raise NotImplementedError()


def aten_reflection_pad3d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64[6]
) -> TensorType:
    raise NotImplementedError()


def aten_relu6(self: FloatType) -> FloatType:
    # TODO(justinchuby): Create a shortcut for creating constants
    zero = op.CastLike(op.Constant(value_float=0.0), self)
    # zero = op.CastLike(0, self)
    return op.Max(self, zero)


def aten_replication_pad1d(self: TensorType, padding: INT64[2]) -> TensorType:
    raise NotImplementedError()


def aten_replication_pad1d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64[2]
) -> TensorType:
    raise NotImplementedError()


def aten_replication_pad2d(self: TensorType, padding: INT64[4]) -> TensorType:
    raise NotImplementedError()


def aten_replication_pad2d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64[4]
) -> TensorType:
    raise NotImplementedError()


def aten_replication_pad3d(self: TensorType, padding: INT64[6]) -> TensorType:
    raise NotImplementedError()


def aten_replication_pad3d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64[6]
) -> TensorType:
    raise NotImplementedError()


def aten_rrelu_with_noise(
    self: TensorType,
    noise: TensorType,
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
    training: bool = False,
    generator: Optional[str] = None,
) -> TensorType:
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
    raise NotImplementedError()


def aten_sigmoid_backward(grad_output: TensorType, output: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_silu(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_silu_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_slow_conv3d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = 1,
    padding: INT64[3] = 0,
) -> TensorType:
    raise NotImplementedError()


def aten_slow_conv3d_forward(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: INT64[3],
) -> TensorType:
    raise NotImplementedError()


def aten_slow_conv_dilated2d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = 1,
    padding: INT64[2] = 0,
    dilation: Sequence[int] = 1,
) -> TensorType:
    raise NotImplementedError()


def aten_slow_conv_dilated3d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = 1,
    padding: INT64[3] = 0,
    dilation: Sequence[int] = 1,
) -> TensorType:
    raise NotImplementedError()


def aten_slow_conv_transpose2d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = 1,
    padding: INT64[2] = 0,
    output_padding: INT64[2] = 0,
    dilation: Sequence[int] = 1,
) -> TensorType:
    raise NotImplementedError()


def aten_slow_conv_transpose3d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = 1,
    padding: INT64[3] = 0,
    output_padding: INT64[3] = 0,
    dilation: Sequence[int] = 1,
) -> TensorType:
    raise NotImplementedError()


def aten_smooth_l1_loss(
    self: TensorType, target: TensorType, reduction: int = "mean", beta: float = 1.0
) -> TensorType:
    raise NotImplementedError()


def aten_smooth_l1_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int, beta: float
) -> TensorType:
    raise NotImplementedError()


def aten_soft_margin_loss(
    self: TensorType, target: TensorType, reduction: int = "mean"
) -> TensorType:
    raise NotImplementedError()


def aten_soft_margin_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int
) -> TensorType:
    raise NotImplementedError()


def aten_softplus(self: TensorType, beta: int = 1, threshold: int = 20) -> TensorType:
    raise NotImplementedError()


def aten_softplus_backward(
    grad_output: TensorType, self: TensorType, beta: float, threshold: float
) -> TensorType:
    raise NotImplementedError()


def aten_softshrink(self: TensorType, lambd: float = 0.5) -> TensorType:
    raise NotImplementedError()


def aten_softshrink_backward(
    grad_output: TensorType, self: TensorType, lambd: float
) -> TensorType:
    raise NotImplementedError()


def aten_tanh_backward(grad_output: TensorType, output: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_thnn_conv2d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = 1,
    padding: Sequence[int] = 0,
) -> TensorType:
    raise NotImplementedError()


def aten_unflatten_dense_tensors(flat: TensorType, tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_upsample_bicubic2d(
    self: TensorType,
    output_size: INT64[2],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_bicubic2d_backward(
    grad_output: TensorType,
    output_size: INT64[2],
    input_size: INT64[4],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_bilinear2d(
    self: TensorType,
    output_size: INT64[2],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_bilinear2d_backward(
    grad_output: TensorType,
    output_size: INT64[2],
    input_size: INT64[4],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_linear1d(
    self: TensorType,
    output_size: INT64[1],
    align_corners: bool,
    scales: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_linear1d_backward(
    grad_output: TensorType,
    output_size: INT64[1],
    input_size: INT64[3],
    align_corners: bool,
    scales: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_nearest1d(
    self: TensorType, output_size: INT64[1], scales: Optional[float] = None
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_nearest1d_backward(
    grad_output: TensorType,
    output_size: INT64[1],
    input_size: INT64[3],
    scales: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_nearest2d(
    self: TensorType,
    output_size: INT64[2],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_nearest2d_backward(
    grad_output: TensorType,
    output_size: INT64[2],
    input_size: INT64[4],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_nearest3d(
    self: TensorType,
    output_size: INT64[3],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_nearest3d_backward(
    grad_output: TensorType,
    output_size: INT64[3],
    input_size: INT64[5],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_trilinear3d(
    self: TensorType,
    output_size: INT64[3],
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_upsample_trilinear3d_backward(
    grad_output: TensorType,
    output_size: INT64[3],
    input_size: INT64[5],
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()
