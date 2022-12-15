# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# mypy: disable-error-code=misc
# mypy: disable-error-code=arg-type
# mypy: disable-error-code=type-arg
# mypy: disable-error-code=valid-type
# mypy: disable-error-code=assignment
"""torch.ops.aten operators under the `nn` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""

# pylint: disable=unused-argument

from __future__ import annotations

from typing import Optional, Sequence

from onnxscript import INT64
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType


def aten_adaptive_avg_pool2d(self: TensorType, output_size: INT64) -> TensorType:
    # adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor

    raise NotImplementedError()


def aten_adaptive_avg_pool3d(self: TensorType, output_size: INT64) -> TensorType:
    # adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor

    raise NotImplementedError()


def aten_adaptive_max_pool2d(
    self: TensorType, output_size: Sequence[int]
) -> tuple[TensorType, TensorType]:
    # adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_adaptive_max_pool2d_backward(
    grad_output: TensorType, self: TensorType, indices: TensorType
) -> TensorType:
    # adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor

    raise NotImplementedError()


def aten_adaptive_max_pool3d(
    self: TensorType, output_size: Sequence[int]
) -> tuple[TensorType, TensorType]:
    # adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_adaptive_max_pool3d_backward(
    grad_output: TensorType, self: TensorType, indices: TensorType
) -> TensorType:
    # adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor

    raise NotImplementedError()


def aten_avg_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> TensorType:
    # avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor

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
    # avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor

    raise NotImplementedError()


def aten_avg_pool3d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> TensorType:
    # avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor

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
    # avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor

    raise NotImplementedError()


def aten_binary_cross_entropy(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
) -> TensorType:
    # binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_binary_cross_entropy_backward(
    grad_output: TensorType,
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
) -> TensorType:
    # binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_celu(self, alpha: float = 1.0):
    # celu(Tensor self, Scalar alpha=1.0) -> Tensor

    raise NotImplementedError()


def aten_col2im(
    self: TensorType,
    output_size: INT64,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
) -> TensorType:
    # col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor

    raise NotImplementedError()


def aten_conv_depthwise3d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: INT64,
    dilation: Sequence[int],
) -> TensorType:
    # conv_depthwise3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, SymInt[3] padding, int[3] dilation) -> Tensor

    raise NotImplementedError()


def aten_cross_entropy_loss(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
    ignore_index: INT64 = -100,
    label_smoothing: float = 0.0,
) -> TensorType:
    # cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, float label_smoothing=0.0) -> Tensor

    raise NotImplementedError()


def aten_elu(
    self,
    alpha: float = 1.0,
    scale: float = 1.0,
    input_scale: float = 1.0,
):
    # elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor

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
    # elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor

    raise NotImplementedError()


def aten_flatten_dense_tensors(tensors: Sequence[TensorType]) -> TensorType:
    # flatten_dense_tensors(Tensor[] tensors) -> Tensor

    raise NotImplementedError()


def aten_fractional_max_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    random_samples: TensorType,
) -> tuple[TensorType, TensorType]:
    # fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_fractional_max_pool2d_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    indices: TensorType,
) -> TensorType:
    # fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor

    raise NotImplementedError()


def aten_fractional_max_pool3d(
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    random_samples: TensorType,
) -> tuple[TensorType, TensorType]:
    # fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_fractional_max_pool3d_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    output_size: Sequence[int],
    indices: TensorType,
) -> TensorType:
    # fractional_max_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices) -> Tensor

    raise NotImplementedError()


def aten_gelu(self: TensorType, approximate: str = "none") -> TensorType:
    # gelu(Tensor self, *, str approximate='none') -> Tensor

    raise NotImplementedError()


def aten_gelu_backward(
    grad_output: TensorType, self: TensorType, approximate: str = "none"
) -> TensorType:
    # gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor

    raise NotImplementedError()


def aten_glu(self: TensorType, dim: int = -1) -> TensorType:
    # glu(Tensor self, int dim=-1) -> Tensor

    raise NotImplementedError()


def aten_glu_backward(grad_output: TensorType, self: TensorType, dim: int) -> TensorType:
    # glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor

    raise NotImplementedError()


def aten_glu_backward_jvp(
    grad_x: TensorType,
    grad_glu: TensorType,
    x: TensorType,
    dgrad_glu: TensorType,
    dx: TensorType,
    dim: int,
) -> TensorType:
    # glu_backward_jvp(Tensor grad_x, Tensor grad_glu, Tensor x, Tensor dgrad_glu, Tensor dx, int dim) -> Tensor

    raise NotImplementedError()


def aten_glu_jvp(glu: TensorType, x: TensorType, dx: TensorType, dim: int) -> TensorType:
    # glu_jvp(Tensor glu, Tensor x, Tensor dx, int dim) -> Tensor

    raise NotImplementedError()


def aten_hardsigmoid(self: TensorType) -> TensorType:
    # hardsigmoid(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_hardsigmoid_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    # hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor

    raise NotImplementedError()


def aten_hardswish(self: TensorType) -> TensorType:
    # hardswish(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_hardswish_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    # hardswish_backward(Tensor grad_output, Tensor self) -> Tensor

    raise NotImplementedError()


def aten_hardtanh(self: TensorType, min_val: float = -1, max_val: float = 1) -> TensorType:
    # hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor

    raise NotImplementedError()


def aten_hardtanh_backward(
    grad_output: TensorType, self: TensorType, min_val: float, max_val: float
) -> TensorType:
    # hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor

    raise NotImplementedError()


def aten_huber_loss(
    self: TensorType, target: TensorType, reduction: int = 1, delta: float = 1.0
) -> TensorType:
    # huber_loss(Tensor self, Tensor target, int reduction=Mean, float delta=1.0) -> Tensor

    raise NotImplementedError()


def aten_huber_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int, delta: float
) -> TensorType:
    # huber_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta) -> Tensor

    raise NotImplementedError()


def aten_im2col(
    self: TensorType,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
) -> TensorType:
    # im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor

    raise NotImplementedError()


def aten_infinitely_differentiable_gelu_backward(
    grad: TensorType, self: TensorType
) -> TensorType:
    # infinitely_differentiable_gelu_backward(Tensor grad, Tensor self) -> Tensor

    raise NotImplementedError()


def aten_l1_loss(self: TensorType, target: TensorType, reduction: int = 1) -> TensorType:
    # l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_leaky_relu(self: TensorType, negative_slope: float = 0.01) -> TensorType:
    # leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor

    raise NotImplementedError()


def aten_leaky_relu_backward(
    grad_output: TensorType, self: TensorType, negative_slope: float, self_is_result: bool
) -> TensorType:
    # leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor

    raise NotImplementedError()


def aten_linear(input, weight, bias=None) -> TensorType:
    # linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor

    # FIXME(justinchuby): Enable the test
    # INVALID_GRAPH : This is an invalid model.
    # In Node, ("", OptionalHasElement, "", -1) : () -> ("output0",) ,
    # Error Node () has input size 0 not in range [min=1, max=1]

    # NOTE: The symbolic function in torch.onnx also uses Gemm in certain cases
    # Optimizers may consider this path and replace it with Gemm
    result = op.MatMul(input, weight)
    if op.OptionalHasElement(bias):
        bias = op.OptionalGetElement(bias)
        result = op.Add(result, bias)  # type: ignore[arg-type]
    return result


def aten_log_sigmoid(self: TensorType) -> TensorType:
    # log_sigmoid(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_log_sigmoid_backward(
    grad_output: TensorType, self: TensorType, buffer: TensorType
) -> TensorType:
    # log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor

    raise NotImplementedError()


def aten_log_sigmoid_forward(self: TensorType) -> tuple[TensorType, TensorType]:
    # log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)

    raise NotImplementedError()


def aten_logit_backward(
    grad_output: TensorType, self: TensorType, eps: Optional[float] = None
) -> TensorType:
    # logit_backward(Tensor grad_output, Tensor self, float? eps=None) -> Tensor

    raise NotImplementedError()


def aten_max_pool2d_with_indices(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> tuple[TensorType, TensorType]:
    # max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)

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
    # max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor

    raise NotImplementedError()


def aten_max_pool3d_with_indices(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    ceil_mode: bool = False,
) -> tuple[TensorType, TensorType]:
    # max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)

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
    # max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor

    raise NotImplementedError()


def aten_max_unpool2d(
    self: TensorType, indices: TensorType, output_size: Sequence[int]
) -> TensorType:
    # max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor

    raise NotImplementedError()


def aten_max_unpool3d(
    self: TensorType,
    indices: TensorType,
    output_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> TensorType:
    # max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor

    raise NotImplementedError()


def aten_mish(self: TensorType) -> TensorType:
    # mish(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_mish_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    # mish_backward(Tensor grad_output, Tensor self) -> Tensor

    raise NotImplementedError()


def aten_mkldnn_linear(
    self: TensorType, weight: TensorType, bias: Optional[TensorType] = None
) -> TensorType:
    # mkldnn_linear(Tensor self, Tensor weight, Tensor? bias=None) -> Tensor

    raise NotImplementedError()


def aten_mkldnn_reorder_conv2d_weight(
    self: TensorType,
    padding: Sequence[int] = (0, 0),
    stride: Sequence[int] = (1, 1),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
) -> TensorType:
    # mkldnn_reorder_conv2d_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1) -> Tensor

    raise NotImplementedError()


def aten_mkldnn_reorder_conv3d_weight(
    self: TensorType,
    padding: Sequence[int] = (0, 0, 0),
    stride: Sequence[int] = (1, 1, 1),
    dilation: Sequence[int] = (1, 1, 1),
    groups: int = 1,
) -> TensorType:
    # mkldnn_reorder_conv3d_weight(Tensor self, int[3] padding=0, int[3] stride=1, int[3] dilation=1, int groups=1) -> Tensor

    raise NotImplementedError()


def aten_mse_loss(self: TensorType, target: TensorType, reduction: int = 1) -> TensorType:
    # mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_mse_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int
) -> TensorType:
    # mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor

    raise NotImplementedError()


def aten_multi_margin_loss(
    self: TensorType,
    target: TensorType,
    p: float = 1,
    margin: float = 1,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
) -> TensorType:
    # multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor

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
    # multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_multilabel_margin_loss(
    self: TensorType, target: TensorType, reduction: int = 1
) -> TensorType:
    # multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_multilabel_margin_loss_backward(
    grad_output: TensorType,
    self: TensorType,
    target: TensorType,
    reduction: int,
    is_target: TensorType,
) -> TensorType:
    # multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor

    raise NotImplementedError()


def aten_multilabel_margin_loss_forward(
    self: TensorType, target: TensorType, reduction: int
) -> tuple[TensorType, TensorType]:
    # multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)

    raise NotImplementedError()


def aten_nll_loss(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
    ignore_index: INT64 = -100,
) -> TensorType:
    # nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor

    raise NotImplementedError()


def aten_nll_loss2d(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
    ignore_index: INT64 = -100,
) -> TensorType:
    # nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor

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
    # nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight) -> Tensor

    raise NotImplementedError()


def aten_nll_loss2d_forward(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType],
    reduction: int,
    ignore_index: INT64,
) -> tuple[TensorType, TensorType]:
    # nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)

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
    # nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight) -> Tensor

    raise NotImplementedError()


def aten_nll_loss_forward(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType],
    reduction: int,
    ignore_index: INT64,
) -> tuple[TensorType, TensorType]:
    # nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)

    raise NotImplementedError()


def aten_nll_loss_nd(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    reduction: int = 1,
    ignore_index: INT64 = -100,
) -> TensorType:
    # nll_loss_nd(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor

    raise NotImplementedError()


def aten_one_hot(self: TensorType, num_classes: int = -1) -> TensorType:
    # one_hot(Tensor self, int num_classes=-1) -> Tensor

    raise NotImplementedError()


def aten_pad(
    self: TensorType, pad: INT64, mode: str = "constant", value: Optional[float] = None
) -> TensorType:
    # pad(Tensor self, SymInt[] pad, str mode="constant", float? value=None) -> Tensor

    raise NotImplementedError()


def aten_pad_sequence(
    sequences: Sequence[TensorType], batch_first: bool = False, padding_value: float = 0.0
) -> TensorType:
    # pad_sequence(Tensor[] sequences, bool batch_first=False, float padding_value=0.0) -> Tensor

    raise NotImplementedError()


def aten_reflection_pad1d(self: TensorType, padding: INT64) -> TensorType:
    # reflection_pad1d(Tensor self, SymInt[2] padding) -> Tensor

    raise NotImplementedError()


def aten_reflection_pad1d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    # reflection_pad1d_backward(Tensor grad_output, Tensor self, SymInt[2] padding) -> Tensor

    raise NotImplementedError()


def aten_reflection_pad2d(self: TensorType, padding: INT64) -> TensorType:
    # reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor

    raise NotImplementedError()


def aten_reflection_pad2d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    # reflection_pad2d_backward(Tensor grad_output, Tensor self, SymInt[4] padding) -> Tensor

    raise NotImplementedError()


def aten_reflection_pad3d(self: TensorType, padding: INT64) -> TensorType:
    # reflection_pad3d(Tensor self, SymInt[6] padding) -> Tensor

    raise NotImplementedError()


def aten_reflection_pad3d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    # reflection_pad3d_backward(Tensor grad_output, Tensor self, SymInt[6] padding) -> Tensor

    raise NotImplementedError()


# TODO(justinchuby): Use TFloat as return type
def aten_relu6(self):
    # relu6(Tensor self) -> Tensor

    return op.Min(op.Relu(self), op.Constant(value_float=6.0))  # type: ignore[arg-type]


def aten_replication_pad1d(self: TensorType, padding: INT64) -> TensorType:
    # replication_pad1d(Tensor self, SymInt[2] padding) -> Tensor

    raise NotImplementedError()


def aten_replication_pad1d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    # replication_pad1d_backward(Tensor grad_output, Tensor self, SymInt[2] padding) -> Tensor

    raise NotImplementedError()


def aten_replication_pad2d(self: TensorType, padding: INT64) -> TensorType:
    # replication_pad2d(Tensor self, SymInt[4] padding) -> Tensor

    raise NotImplementedError()


def aten_replication_pad2d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    # replication_pad2d_backward(Tensor grad_output, Tensor self, SymInt[4] padding) -> Tensor

    raise NotImplementedError()


def aten_replication_pad3d(self: TensorType, padding: INT64) -> TensorType:
    # replication_pad3d(Tensor self, SymInt[6] padding) -> Tensor

    raise NotImplementedError()


def aten_replication_pad3d_backward(
    grad_output: TensorType, self: TensorType, padding: INT64
) -> TensorType:
    # replication_pad3d_backward(Tensor grad_output, Tensor self, SymInt[6] padding) -> Tensor

    raise NotImplementedError()


def aten_rrelu_with_noise(
    self: TensorType,
    noise: TensorType,
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
    training: bool = False,
    generator: Optional[str] = None,
) -> TensorType:
    # rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor

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
    # rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result) -> Tensor

    raise NotImplementedError()


def aten_sigmoid_backward(grad_output: TensorType, output: TensorType) -> TensorType:
    # sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor

    raise NotImplementedError()


def aten_silu(self: TensorType) -> TensorType:
    # silu(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_silu_backward(grad_output: TensorType, self: TensorType) -> TensorType:
    # silu_backward(Tensor grad_output, Tensor self) -> Tensor

    raise NotImplementedError()


def aten_slow_conv3d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1, 1, 1),
    padding: INT64 = (0, 0, 0),
) -> TensorType:
    # slow_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, SymInt[3] padding=0) -> Tensor

    raise NotImplementedError()


def aten_slow_conv3d_forward(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: INT64,
) -> TensorType:
    # slow_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, SymInt[3] padding) -> Tensor

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
    # slow_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, SymInt[2] padding=0, int[2] dilation=1) -> Tensor

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
    # slow_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, SymInt[3] padding=0, int[3] dilation=1) -> Tensor

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
    # slow_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, SymInt[2] padding=0, SymInt[2] output_padding=0, int[2] dilation=1) -> Tensor

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
    # slow_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, SymInt[3] padding=0, SymInt[3] output_padding=0, int[3] dilation=1) -> Tensor

    raise NotImplementedError()


def aten_smooth_l1_loss(
    self: TensorType, target: TensorType, reduction: int = 1, beta: float = 1.0
) -> TensorType:
    # smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean, float beta=1.0) -> Tensor

    raise NotImplementedError()


def aten_smooth_l1_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int, beta: float
) -> TensorType:
    # smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta) -> Tensor

    raise NotImplementedError()


def aten_soft_margin_loss(
    self: TensorType, target: TensorType, reduction: int = 1
) -> TensorType:
    # soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_soft_margin_loss_backward(
    grad_output: TensorType, self: TensorType, target: TensorType, reduction: int
) -> TensorType:
    # soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor

    raise NotImplementedError()


def aten_softplus(self: TensorType, beta: float = 1, threshold: float = 20) -> TensorType:
    # softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor

    raise NotImplementedError()


def aten_softplus_backward(
    grad_output: TensorType, self: TensorType, beta: float, threshold: float
) -> TensorType:
    # softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold) -> Tensor

    raise NotImplementedError()


def aten_softshrink(self: TensorType, lambd: float = 0.5) -> TensorType:
    # softshrink(Tensor self, Scalar lambd=0.5) -> Tensor

    raise NotImplementedError()


def aten_softshrink_backward(
    grad_output: TensorType, self: TensorType, lambd: float
) -> TensorType:
    # softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor

    raise NotImplementedError()


def aten_tanh_backward(grad_output: TensorType, output: TensorType) -> TensorType:
    # tanh_backward(Tensor grad_output, Tensor output) -> Tensor

    raise NotImplementedError()


def aten_thnn_conv2d(
    self: TensorType,
    weight: TensorType,
    kernel_size: Sequence[int],
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
) -> TensorType:
    # thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0) -> Tensor

    raise NotImplementedError()


def aten_unflatten_dense_tensors(
    flat: TensorType, tensors: Sequence[TensorType]
) -> TensorType:
    # unflatten_dense_tensors(Tensor flat, Tensor[] tensors) -> Tensor[]

    raise NotImplementedError()


def aten_upsample_bicubic2d(
    self: TensorType,
    output_size: INT64,
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    # upsample_bicubic2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_bicubic2d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    # upsample_bicubic2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_bilinear2d(
    self: TensorType,
    output_size: INT64,
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    # upsample_bilinear2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_bilinear2d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    # upsample_bilinear2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_linear1d(
    self: TensorType, output_size: INT64, align_corners: bool, scales: Optional[float] = None
) -> TensorType:
    # upsample_linear1d(Tensor self, SymInt[1] output_size, bool align_corners, float? scales=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_linear1d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    align_corners: bool,
    scales: Optional[float] = None,
) -> TensorType:
    # upsample_linear1d_backward(Tensor grad_output, SymInt[1] output_size, SymInt[3] input_size, bool align_corners, float? scales=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_nearest1d(
    self: TensorType, output_size: INT64, scales: Optional[float] = None
) -> TensorType:
    # upsample_nearest1d(Tensor self, SymInt[1] output_size, float? scales=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_nearest1d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    scales: Optional[float] = None,
) -> TensorType:
    # upsample_nearest1d_backward(Tensor grad_output, SymInt[1] output_size, SymInt[3] input_size, float? scales=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_nearest2d(
    self: TensorType,
    output_size: INT64,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    # upsample_nearest2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_nearest2d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    # upsample_nearest2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_nearest3d(
    self: TensorType,
    output_size: INT64,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    # upsample_nearest3d(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_nearest3d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    # upsample_nearest3d_backward(Tensor grad_output, SymInt[3] output_size, SymInt[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_trilinear3d(
    self: TensorType,
    output_size: INT64,
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    # upsample_trilinear3d(Tensor self, SymInt[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor

    raise NotImplementedError()


def aten_upsample_trilinear3d_backward(
    grad_output: TensorType,
    output_size: INT64,
    input_size: INT64,
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> TensorType:
    # upsample_trilinear3d_backward(Tensor grad_output, SymInt[3] output_size, SymInt[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor

    raise NotImplementedError()
