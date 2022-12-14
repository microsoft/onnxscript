# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# mypy: disable-error-code=misc
# mypy: disable-error-code=arg-type
# mypy: disable-error-code=type-arg
# mypy: disable-error-code=valid-type
# mypy: disable-error-code=assignment
"""torch.ops.aten operators under the `core` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from onnxscript import BOOL, INT64
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType


def aten_abs(self):
    # abs(Tensor self) -> Tensor

    return op.Abs(self)


def aten_acos(self):
    # acos(Tensor self) -> Tensor

    return op.Acos(self)


def aten_acosh(self):
    # acosh(Tensor self) -> Tensor

    return op.Acosh(self)


def aten_adaptive_avg_pool1d(self: TensorType, output_size: Sequence[int]) -> TensorType:
    # adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor

    raise NotImplementedError()


def aten_adaptive_max_pool1d(
    self: TensorType, output_size: Sequence[int]
) -> tuple[TensorType, TensorType]:
    # adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_add(self, other, alpha: float = 1) -> TensorType:
    # add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    if alpha != 1:
        other = op.Mul(other, alpha)  # type: ignore[arg-type]
    return op.Add(self, other)


def aten_addbmm(
    self: TensorType, batch1: TensorType, batch2: TensorType, beta: float = 1, alpha: float = 1
) -> TensorType:
    # addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

    raise NotImplementedError()


def aten_addcdiv(
    self: TensorType, tensor1: TensorType, tensor2: TensorType, value: float = 1
) -> TensorType:
    # addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

    raise NotImplementedError()


def aten_addcmul(
    self: TensorType, tensor1: TensorType, tensor2: TensorType, value: float = 1
) -> TensorType:
    # addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

    raise NotImplementedError()


def aten_addmm(self, mat1, mat2, beta: float = 1, alpha: float = 1):
    # addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

    mat1_mat2 = op.MatMul(mat1, mat2)
    scaled_mat1_mat2 = op.Mul(mat1_mat2, alpha)  # type: ignore[arg-type]
    scaled_self = op.Mul(self, beta)  # type: ignore[arg-type]
    return op.Add(scaled_self, scaled_mat1_mat2)  # type: ignore[arg-type]


def aten_addmv(
    self: TensorType, mat: TensorType, vec: TensorType, beta: float = 1, alpha: float = 1
) -> TensorType:
    # addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor

    raise NotImplementedError()


def aten_addr(
    self: TensorType, vec1: TensorType, vec2: TensorType, beta: float = 1, alpha: float = 1
) -> TensorType:
    # addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

    raise NotImplementedError()


def aten_adjoint(self: TensorType) -> TensorType:
    # adjoint(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_affine_grid_generator(
    theta: TensorType, size: Sequence[int], align_corners: bool
) -> TensorType:
    # affine_grid_generator(Tensor theta, int[] size, bool align_corners) -> Tensor

    raise NotImplementedError()


def aten_affine_grid_generator_backward(
    grad: TensorType, size: Sequence[int], align_corners: bool
) -> TensorType:
    # affine_grid_generator_backward(Tensor grad, int[] size, bool align_corners) -> Tensor

    raise NotImplementedError()


def aten_alias(self: TensorType) -> TensorType:
    # alias(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_alias_copy(self: TensorType) -> TensorType:
    # alias_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_align_as(self: TensorType, other: TensorType) -> TensorType:
    # align_as(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_align_tensors(tensors: Sequence[TensorType]) -> TensorType:
    # align_tensors(Tensor[] tensors) -> Tensor[]

    raise NotImplementedError()


def aten_align_to(self: TensorType, names: Sequence[str]) -> TensorType:
    # align_to(Tensor(a) self, Dimname[] names) -> Tensor(a)

    raise NotImplementedError()


def aten_all(self: TensorType) -> TensorType:
    # all(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_allclose(
    self: TensorType,
    other: TensorType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    # allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool

    raise NotImplementedError()


def aten_alpha_dropout(input: TensorType, p: float, train: bool) -> TensorType:
    # alpha_dropout(Tensor input, float p, bool train) -> Tensor

    raise NotImplementedError()


def aten_amax(
    self: TensorType, dim: Optional[Sequence[int]] = None, keepdim: bool = False
) -> TensorType:
    # amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor

    raise NotImplementedError()


def aten_amin(
    self: TensorType, dim: Optional[Sequence[int]] = None, keepdim: bool = False
) -> TensorType:
    # amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor

    raise NotImplementedError()


def aten_aminmax(
    self: TensorType, dim: Optional[int] = None, keepdim: bool = False
) -> tuple[TensorType, TensorType]:
    # aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)

    raise NotImplementedError()


def aten_and(self: TensorType, other: TensorType) -> TensorType:
    # __and__.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_angle(self: TensorType) -> TensorType:
    # angle(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_any(self: TensorType) -> TensorType:
    # any(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_arange(end: float) -> TensorType:
    # arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_arccos(self: TensorType) -> TensorType:
    # arccos(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_arccosh(self: TensorType) -> TensorType:
    # arccosh(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_arcsin(self: TensorType) -> TensorType:
    # arcsin(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_arcsinh(self: TensorType) -> TensorType:
    # arcsinh(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_arctan(self: TensorType) -> TensorType:
    # arctan(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_arctan2(self: TensorType, other: TensorType) -> TensorType:
    # arctan2(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_arctanh(self: TensorType) -> TensorType:
    # arctanh(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_argmax(
    self: TensorType, dim: Optional[int] = None, keepdim: bool = False
) -> TensorType:
    # argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor

    raise NotImplementedError()


def aten_argmin(
    self: TensorType, dim: Optional[int] = None, keepdim: bool = False
) -> TensorType:
    # argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor

    raise NotImplementedError()


def aten_argsort(self: TensorType, dim: int = -1, descending: bool = False) -> TensorType:
    # argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor

    raise NotImplementedError()


def aten_argwhere(self: TensorType) -> TensorType:
    # argwhere(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_as_strided(
    self: TensorType, size: INT64, stride: INT64, storage_offset: Optional[INT64] = None
) -> TensorType:
    # as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)

    raise NotImplementedError()


def aten_as_strided_copy(
    self: TensorType, size: INT64, stride: INT64, storage_offset: Optional[INT64] = None
) -> TensorType:
    # as_strided_copy(Tensor self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor

    raise NotImplementedError()


def aten_as_strided_scatter(
    self: TensorType,
    src: TensorType,
    size: INT64,
    stride: INT64,
    storage_offset: Optional[INT64] = None,
) -> TensorType:
    # as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor

    raise NotImplementedError()


def aten_asin(self: TensorType) -> TensorType:
    # asin(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_asinh(self: TensorType) -> TensorType:
    # asinh(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_atan(self: TensorType) -> TensorType:
    # atan(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_atan2(self: TensorType, other: TensorType) -> TensorType:
    # atan2(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_atanh(self: TensorType) -> TensorType:
    # atanh(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_atleast_1d(self: TensorType) -> TensorType:
    # atleast_1d(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_atleast_2d(self: TensorType) -> TensorType:
    # atleast_2d(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_atleast_3d(self: TensorType) -> TensorType:
    # atleast_3d(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_avg_pool1d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0,),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> TensorType:
    # avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor

    raise NotImplementedError()


def aten_baddbmm(
    self: TensorType, batch1: TensorType, batch2: TensorType, beta: float = 1, alpha: float = 1
) -> TensorType:
    # baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

    raise NotImplementedError()


def aten_bartlett_window(window_length: int) -> TensorType:
    # bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_batch_norm(
    input: TensorType,
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    training: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool,
) -> TensorType:
    # batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor

    raise NotImplementedError()


def aten_batch_norm_backward_elemt(
    grad_out: TensorType,
    input: TensorType,
    mean: TensorType,
    invstd: TensorType,
    weight: Optional[TensorType],
    mean_dy: TensorType,
    mean_dy_xmu: TensorType,
    count: TensorType,
) -> TensorType:
    # batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu, Tensor count) -> Tensor

    raise NotImplementedError()


def aten_batch_norm_backward_reduce(
    grad_out: TensorType,
    input: TensorType,
    mean: TensorType,
    invstd: TensorType,
    weight: Optional[TensorType],
    input_g: bool,
    weight_g: bool,
    bias_g: bool,
) -> tuple[TensorType, TensorType, TensorType, TensorType]:
    # batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_batch_norm_elemt(
    input: TensorType,
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    mean: TensorType,
    invstd: TensorType,
    eps: float,
) -> TensorType:
    # batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor

    raise NotImplementedError()


def aten_batch_norm_gather_stats(
    input: TensorType,
    mean: TensorType,
    invstd: TensorType,
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    momentum: float,
    eps: float,
    count: int,
) -> tuple[TensorType, TensorType]:
    # batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_batch_norm_gather_stats_with_counts(
    input: TensorType,
    mean: TensorType,
    invstd: TensorType,
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    momentum: float,
    eps: float,
    counts: TensorType,
) -> tuple[TensorType, TensorType]:
    # batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_batch_norm_stats(input: TensorType, eps: float) -> tuple[TensorType, TensorType]:
    # batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_batch_norm_update_stats(
    input: TensorType,
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    momentum: float,
) -> tuple[TensorType, TensorType]:
    # batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_bernoulli(self: TensorType, generator: Optional[str] = None) -> TensorType:
    # bernoulli(Tensor self, *, Generator? generator=None) -> Tensor

    raise NotImplementedError()


def aten_bilinear(
    input1: TensorType,
    input2: TensorType,
    weight: TensorType,
    bias: Optional[TensorType] = None,
) -> TensorType:
    # bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None) -> Tensor

    raise NotImplementedError()


def aten_binary_cross_entropy_with_logits(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    pos_weight: Optional[TensorType] = None,
    reduction: int = 1,
) -> TensorType:
    # binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_bincount(
    self: TensorType, weights: Optional[TensorType] = None, minlength: int = 0
) -> TensorType:
    # bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor

    raise NotImplementedError()


def aten_binomial(
    count: TensorType, prob: TensorType, generator: Optional[str] = None
) -> TensorType:
    # binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor

    raise NotImplementedError()


def aten_bitwise_and(self: TensorType, other: TensorType) -> TensorType:
    # bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_bitwise_left_shift(self: TensorType, other: TensorType) -> TensorType:
    # bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_bitwise_not(self: TensorType) -> TensorType:
    # bitwise_not(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_bitwise_or(self: TensorType, other: TensorType) -> TensorType:
    # bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_bitwise_right_shift(self: TensorType, other: TensorType) -> TensorType:
    # bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_bitwise_xor(self: TensorType, other: TensorType) -> TensorType:
    # bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_blackman_window(window_length: int) -> TensorType:
    # blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_block_diag(tensors: Sequence[TensorType]) -> TensorType:
    # block_diag(Tensor[] tensors) -> Tensor

    raise NotImplementedError()


def aten_bmm(self, mat2):
    # bmm(Tensor self, Tensor mat2) -> Tensor

    return op.MatMul(self, mat2)


def aten_broadcast_tensors(tensors: Sequence[TensorType]) -> TensorType:
    # broadcast_tensors(Tensor[] tensors) -> Tensor[]

    raise NotImplementedError()


def aten_broadcast_to(self: TensorType, size: INT64) -> TensorType:
    # broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)

    raise NotImplementedError()


def aten_bucketize(
    self: TensorType, boundaries: TensorType, out_int32: bool = False, right: bool = False
) -> TensorType:
    # bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor

    raise NotImplementedError()


def aten_can_cast(from_: int, to: int) -> bool:
    # can_cast(ScalarType from, ScalarType to) -> bool

    raise NotImplementedError()


def aten_cartesian_prod(tensors: Sequence[TensorType]) -> TensorType:
    # cartesian_prod(Tensor[] tensors) -> Tensor

    raise NotImplementedError()


def aten_cat(tensors: Sequence[TensorType], dim: int = 0) -> TensorType:
    # cat(Tensor[] tensors, int dim=0) -> Tensor

    raise NotImplementedError()


def aten_ccol_indices(self: TensorType) -> TensorType:
    # ccol_indices(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_ccol_indices_copy(self: TensorType) -> TensorType:
    # ccol_indices_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_cdist(
    x1: TensorType, x2: TensorType, p: float = 2, compute_mode: Optional[int] = None
) -> TensorType:
    # cdist(Tensor x1, Tensor x2, float p=2, int? compute_mode=None) -> Tensor

    raise NotImplementedError()


def aten_ceil(self: TensorType) -> TensorType:
    # ceil(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_celu(self: TensorType, alpha: float = 1.0) -> TensorType:
    # celu(Tensor self, Scalar alpha=1.0) -> Tensor

    raise NotImplementedError()


def aten_chain_matmul(matrices: Sequence[TensorType]) -> TensorType:
    # chain_matmul(Tensor[] matrices) -> Tensor

    raise NotImplementedError()


def aten_chalf(self: TensorType, memory_format: Optional[str] = None) -> TensorType:
    # chalf(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor

    raise NotImplementedError()


def aten_channel_shuffle(self: TensorType, groups: int) -> TensorType:
    # channel_shuffle(Tensor self, int groups) -> Tensor

    raise NotImplementedError()


def aten_cholesky(self: TensorType, upper: bool = False) -> TensorType:
    # cholesky(Tensor self, bool upper=False) -> Tensor

    raise NotImplementedError()


def aten_cholesky_inverse(self: TensorType, upper: bool = False) -> TensorType:
    # cholesky_inverse(Tensor self, bool upper=False) -> Tensor

    raise NotImplementedError()


def aten_cholesky_solve(
    self: TensorType, input2: TensorType, upper: bool = False
) -> TensorType:
    # cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor

    raise NotImplementedError()


def aten_choose_qparams_optimized(
    input: TensorType, numel: int, n_bins: int, ratio: float, bit_width: int
) -> tuple[TensorType, TensorType]:
    # choose_qparams_optimized(Tensor input, int numel, int n_bins, float ratio, int bit_width) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_chunk(self: TensorType, chunks: int, dim: int = 0) -> TensorType:
    # chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]

    raise NotImplementedError()


def aten_clamp(self: TensorType, min_=None, max_=None) -> TensorType:
    # clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor

    # TODO(justinchuby): Handle integer inputs
    # FIXME(justinchuby): Enable test for this after None values are supported
    # TODO(justinchuby): If min is greater than max torch.clamp(..., min, max)
    # sets all elements in input to the value of max.
    if op.OptionalHasElement(min_):
        min_ = op.OptionalGetElement(min_)
        min_clamp = op.CastLike(min_, self)  # type: ignore[arg-type]
    else:
        min_clamp = op.Constant(value_float=float("-inf"))

    if op.OptionalHasElement(max_):
        max_ = op.OptionalGetElement(max_)
        max_clamp = op.CastLike(max_, self)  # type: ignore[arg-type]
    else:
        max_clamp = op.Constant(value_float=float("inf"))

    # Enforce the lower and upper bounds
    clamped = op.Max(op.Min(self, max_clamp), min_clamp)  # type: ignore[arg-type]
    return clamped


def aten_clamp_max_scalar(self, max_):
    # clamp_max(Tensor self, Scalar max) -> Tensor

    max_ = op.CastLike(max_, self)
    return op.Clip(self, None, max_)


def aten_clamp_max_tensor(self, max_):
    # clamp_max(Tensor self, Scalar max) -> Tensor

    return op.Min(self, max_)


def aten_clamp_min_scalar(self, min_):
    # clamp_min(Tensor self, Scalar min) -> Tensor
    # NOTE: min_ is a rank 0 tensor.
    # TODO(justinchuby): Specify the type constraints.
    min_ = op.CastLike(min_, self)
    return op.Clip(self, min_, None)


def aten_clamp_min_tensor(self, min_):
    # clamp_min(Tensor self, Tensor min) -> Tensor
    # TODO(justinchuby): Specify the type constraints.
    return op.Max(self, min_)


def aten_clip(
    self: TensorType, min: Optional[float] = None, max: Optional[float] = None
) -> TensorType:
    # clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor

    raise NotImplementedError()


def aten_clone(self: TensorType, memory_format: Optional[str] = None) -> TensorType:
    # clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor

    raise NotImplementedError()


def aten_coalesce(self: TensorType) -> TensorType:
    # coalesce(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_col_indices(self: TensorType) -> TensorType:
    # col_indices(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_col_indices_copy(self: TensorType) -> TensorType:
    # col_indices_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_column_stack(tensors: Sequence[TensorType]) -> TensorType:
    # column_stack(Tensor[] tensors) -> Tensor

    raise NotImplementedError()


def aten_combinations(
    self: TensorType, r: int = 2, with_replacement: bool = False
) -> TensorType:
    # combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor

    raise NotImplementedError()


def aten_complex(real: TensorType, imag: TensorType) -> TensorType:
    # complex(Tensor real, Tensor imag) -> Tensor

    raise NotImplementedError()


def aten_concat(tensors: Sequence[TensorType], dim: int = 0) -> TensorType:
    # concat(Tensor[] tensors, int dim=0) -> Tensor

    raise NotImplementedError()


def aten_concatenate(tensors: Sequence[TensorType], dim: int = 0) -> TensorType:
    # concatenate(Tensor[] tensors, int dim=0) -> Tensor

    raise NotImplementedError()


def aten_conj(self: TensorType) -> TensorType:
    # conj(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_conj_physical(self: TensorType) -> TensorType:
    # conj_physical(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_constant_pad_nd(self: TensorType, pad: INT64, value: float = 0) -> TensorType:
    # constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor

    raise NotImplementedError()


def aten_contiguous(self: TensorType, memory_format: str = "contiguous_format") -> TensorType:
    # contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)

    raise NotImplementedError()


def aten_conv1d(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    groups: int = 1,
) -> TensorType:
    # conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor

    raise NotImplementedError()


def aten_conv2d(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
) -> TensorType:
    # conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor

    raise NotImplementedError()


def aten_conv3d(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1, 1, 1),
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    groups: int = 1,
) -> TensorType:
    # conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor

    raise NotImplementedError()


def aten_conv_tbc(
    self: TensorType, weight: TensorType, bias: TensorType, pad: int = 0
) -> TensorType:
    # conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor

    raise NotImplementedError()


def aten_conv_tbc_backward(
    self: TensorType, input: TensorType, weight: TensorType, bias: TensorType, pad: int
) -> tuple[TensorType, TensorType, TensorType]:
    # conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_conv_transpose1d(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    output_padding: Sequence[int] = (0,),
    groups: int = 1,
    dilation: Sequence[int] = (1,),
) -> TensorType:
    # conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor

    raise NotImplementedError()


def aten_convolution(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: INT64,
    dilation: Sequence[int],
    transposed: bool,
    output_padding: INT64,
    groups: int,
) -> TensorType:
    # convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups) -> Tensor

    raise NotImplementedError()


def aten_convolution_backward(
    grad_output: TensorType,
    input: TensorType,
    weight: TensorType,
    bias_sizes: Optional[INT64],
    stride: Sequence[int],
    padding: INT64,
    dilation: Sequence[int],
    transposed: bool,
    output_padding: INT64,
    groups: int,
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType, TensorType]:
    # convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_convolution_backward_overrideable(
    grad_output: TensorType,
    input: TensorType,
    weight: TensorType,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType, TensorType]:
    # convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)

    raise NotImplementedError()


def aten_convolution_overrideable(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
) -> TensorType:
    # convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor

    raise NotImplementedError()


def aten_copy(self: TensorType, src: TensorType, non_blocking: bool = False) -> TensorType:
    # copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor

    raise NotImplementedError()


def aten_copysign(self: TensorType, other: TensorType) -> TensorType:
    # copysign.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_corrcoef(self: TensorType) -> TensorType:
    # corrcoef(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_cos(self: TensorType) -> TensorType:
    # cos(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_cosh(self: TensorType) -> TensorType:
    # cosh(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_cosine_embedding_loss(
    input1: TensorType,
    input2: TensorType,
    target: TensorType,
    margin: float = 0.0,
    reduction: int = 1,
) -> TensorType:
    # cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_cosine_similarity(
    x1: TensorType, x2: TensorType, dim: int = 1, eps: float = 1e-08
) -> TensorType:
    # cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor

    raise NotImplementedError()


def aten_count_nonzero(self: TensorType, dim: Optional[int] = None) -> TensorType:
    # count_nonzero(Tensor self, int? dim=None) -> Tensor

    raise NotImplementedError()


def aten_cov(
    self: TensorType,
    correction: int = 1,
    fweights: Optional[TensorType] = None,
    aweights: Optional[TensorType] = None,
) -> TensorType:
    # cov(Tensor self, *, int correction=1, Tensor? fweights=None, Tensor? aweights=None) -> Tensor

    raise NotImplementedError()


def aten_cross(self: TensorType, other: TensorType, dim: Optional[int] = None) -> TensorType:
    # cross(Tensor self, Tensor other, int? dim=None) -> Tensor

    raise NotImplementedError()


def aten_crow_indices(self: TensorType) -> TensorType:
    # crow_indices(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_crow_indices_copy(self: TensorType) -> TensorType:
    # crow_indices_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_ctc_loss(
    log_probs: TensorType,
    targets: TensorType,
    input_lengths: TensorType,
    target_lengths: TensorType,
    blank: int = 0,
    reduction: int = 1,
    zero_infinity: bool = False,
) -> TensorType:
    # ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor

    raise NotImplementedError()


def aten_cudnn_affine_grid_generator(
    theta: TensorType, N: int, C: int, H: int, W: int
) -> TensorType:
    # cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid

    raise NotImplementedError()


def aten_cudnn_affine_grid_generator_backward(
    grad: TensorType, N: int, C: int, H: int, W: int
) -> TensorType:
    # cudnn_affine_grid_generator_backward(Tensor grad, int N, int C, int H, int W) -> Tensor grad_theta

    raise NotImplementedError()


def aten_cudnn_batch_norm(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
) -> tuple[TensorType, TensorType, TensorType, TensorType]:
    # cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_cudnn_batch_norm_backward(
    input: TensorType,
    grad_output: TensorType,
    weight: TensorType,
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    save_mean: Optional[TensorType],
    save_var: Optional[TensorType],
    epsilon: float,
    reserveSpace: TensorType,
) -> tuple[TensorType, TensorType, TensorType]:
    # cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_cudnn_convolution(
    self: TensorType,
    weight: TensorType,
    padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    benchmark: bool,
    deterministic: bool,
    allow_tf32: bool,
) -> TensorType:
    # cudnn_convolution(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor

    raise NotImplementedError()


def aten_cudnn_convolution_add_relu(
    self: TensorType,
    weight: TensorType,
    z: TensorType,
    alpha: Optional[float],
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> TensorType:
    # cudnn_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor

    raise NotImplementedError()


def aten_cudnn_convolution_relu(
    self: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> TensorType:
    # cudnn_convolution_relu(Tensor self, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor

    raise NotImplementedError()


def aten_cudnn_convolution_transpose(
    self: TensorType,
    weight: TensorType,
    padding: Sequence[int],
    output_padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    benchmark: bool,
    deterministic: bool,
    allow_tf32: bool,
) -> TensorType:
    # cudnn_convolution_transpose(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor

    raise NotImplementedError()


def aten_cudnn_grid_sampler(self: TensorType, grid: TensorType) -> TensorType:
    # cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output

    raise NotImplementedError()


def aten_cudnn_grid_sampler_backward(
    self: TensorType, grid: TensorType, grad_output: TensorType
) -> tuple[TensorType, TensorType]:
    # cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor grad_self, Tensor grad_grid)

    raise NotImplementedError()


def aten_cudnn_is_acceptable(self: TensorType) -> bool:
    # cudnn_is_acceptable(Tensor self) -> bool

    raise NotImplementedError()


def aten_cummax(self: TensorType, dim: int) -> tuple[TensorType, TensorType]:
    # cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)

    raise NotImplementedError()


def aten_cummaxmin_backward(
    grad: TensorType, input: TensorType, indices: TensorType, dim: int
) -> TensorType:
    # cummaxmin_backward(Tensor grad, Tensor input, Tensor indices, int dim) -> Tensor

    raise NotImplementedError()


def aten_cummin(self: TensorType, dim: int) -> tuple[TensorType, TensorType]:
    # cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)

    raise NotImplementedError()


def aten_cumprod(self: TensorType, dim: int, dtype: Optional[int] = None) -> TensorType:
    # cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor

    raise NotImplementedError()


def aten_cumprod_backward(
    grad: TensorType, input: TensorType, dim: int, output: TensorType
) -> TensorType:
    # cumprod_backward(Tensor grad, Tensor input, int dim, Tensor output) -> Tensor

    raise NotImplementedError()


def aten_cumsum(self: TensorType, dim: int, dtype: Optional[int] = None) -> TensorType:
    # cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor

    raise NotImplementedError()


def aten_data(self: TensorType) -> TensorType:
    # data(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_deg2rad(self: TensorType) -> TensorType:
    # deg2rad(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_dense_dim(self: TensorType) -> int:
    # dense_dim(Tensor self) -> int

    raise NotImplementedError()


def aten_det(self: TensorType) -> TensorType:
    # det(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_detach(self: TensorType) -> TensorType:
    # detach(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_detach_copy(self: TensorType) -> TensorType:
    # detach_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_diag(self: TensorType, diagonal: int = 0) -> TensorType:
    # diag(Tensor self, int diagonal=0) -> Tensor

    raise NotImplementedError()


def aten_diag_embed(
    self: TensorType, offset: int = 0, dim1: int = -2, dim2: int = -1
) -> TensorType:
    # diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor

    raise NotImplementedError()


def aten_diagflat(self: TensorType, offset: int = 0) -> TensorType:
    # diagflat(Tensor self, int offset=0) -> Tensor

    raise NotImplementedError()


def aten_diagonal(
    self: TensorType, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> TensorType:
    # diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)

    raise NotImplementedError()


def aten_diagonal_backward(
    grad_output: TensorType, input_sizes: INT64, offset: int, dim1: int, dim2: int
) -> TensorType:
    # diagonal_backward(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2) -> Tensor

    raise NotImplementedError()


def aten_diagonal_copy(
    self: TensorType, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> TensorType:
    # diagonal_copy(Tensor self, int offset=0, int dim1=0, int dim2=1) -> Tensor

    raise NotImplementedError()


def aten_diagonal_scatter(
    self: TensorType, src: TensorType, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> TensorType:
    # diagonal_scatter(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1) -> Tensor

    raise NotImplementedError()


def aten_diff(
    self: TensorType,
    n: int = 1,
    dim: int = -1,
    prepend: Optional[TensorType] = None,
    append: Optional[TensorType] = None,
) -> TensorType:
    # diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor

    raise NotImplementedError()


def aten_digamma(self: TensorType) -> TensorType:
    # digamma(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_dist(self: TensorType, other: TensorType, p: float = 2) -> TensorType:
    # dist(Tensor self, Tensor other, Scalar p=2) -> Tensor

    raise NotImplementedError()


def aten_div(self: TensorType, other: TensorType) -> TensorType:
    # div.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_divide(self: TensorType, other: TensorType) -> TensorType:
    # divide.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_dot(self: TensorType, tensor: TensorType) -> TensorType:
    # dot(Tensor self, Tensor tensor) -> Tensor

    raise NotImplementedError()


def aten_dropout(input: TensorType, p: float, train: bool) -> TensorType:
    # dropout(Tensor input, float p, bool train) -> Tensor

    raise NotImplementedError()


def aten_dstack(tensors: Sequence[TensorType]) -> TensorType:
    # dstack(Tensor[] tensors) -> Tensor

    raise NotImplementedError()


def aten_einsum(
    equation: str, tensors: Sequence[TensorType], path: Optional[int] = None
) -> TensorType:
    # einsum(str equation, Tensor[] tensors, *, int[]? path=None) -> Tensor

    raise NotImplementedError()


def aten_embedding(
    weight: TensorType,
    indices: TensorType,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> TensorType:
    # embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor

    raise NotImplementedError()


def aten_embedding_backward(
    grad: TensorType,
    indices: TensorType,
    num_weights: INT64,
    padding_idx: int,
    scale_grad_by_freq: bool,
    sparse: bool,
) -> TensorType:
    # embedding_backward(Tensor grad, Tensor indices, SymInt num_weights, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor

    raise NotImplementedError()


def aten_embedding_bag(
    weight: TensorType,
    indices: TensorType,
    offsets: TensorType,
    scale_grad_by_freq: bool = False,
    mode: int = 0,
    sparse: bool = False,
    per_sample_weights: Optional[TensorType] = None,
    include_last_offset: bool = False,
) -> tuple[TensorType, TensorType, TensorType, TensorType]:
    # embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_embedding_dense_backward(
    grad_output: TensorType,
    indices: TensorType,
    num_weights: INT64,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> TensorType:
    # embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor

    raise NotImplementedError()


def aten_embedding_sparse_backward(
    grad: TensorType,
    indices: TensorType,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> TensorType:
    # embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor

    raise NotImplementedError()


def aten_empty_like(self: TensorType, memory_format: Optional[str] = None) -> TensorType:
    # empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    raise NotImplementedError()


def aten_empty_quantized(
    size: Sequence[int], qtensor: TensorType, memory_format: Optional[str] = None
) -> TensorType:
    # empty_quantized(int[] size, Tensor qtensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    raise NotImplementedError()


def aten_empty_strided(size: INT64, stride: INT64) -> TensorType:
    # empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_eq(self: TensorType, other: TensorType) -> TensorType:
    # eq.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_equal(self: TensorType, other: TensorType) -> bool:
    # equal(Tensor self, Tensor other) -> bool

    raise NotImplementedError()


def aten_erf(self: TensorType) -> TensorType:
    # erf(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_erfc(self: TensorType) -> TensorType:
    # erfc(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_erfinv(self: TensorType) -> TensorType:
    # erfinv(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_exp(self: TensorType) -> TensorType:
    # exp(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_exp2(self: TensorType) -> TensorType:
    # exp2(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_expand(self: TensorType, size: INT64, implicit: bool = False) -> TensorType:
    # expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)

    raise NotImplementedError()


def aten_expand_as(self: TensorType, other: TensorType) -> TensorType:
    # expand_as(Tensor(a) self, Tensor other) -> Tensor(a)

    raise NotImplementedError()


def aten_expand_copy(self: TensorType, size: INT64, implicit: bool = False) -> TensorType:
    # expand_copy(Tensor self, SymInt[] size, *, bool implicit=False) -> Tensor

    raise NotImplementedError()


def aten_expm1(self: TensorType) -> TensorType:
    # expm1(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_eye(n: int) -> TensorType:
    # eye(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_fake_quantize_per_channel_affine(
    self: TensorType,
    scale: TensorType,
    zero_point: TensorType,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> TensorType:
    # fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor

    raise NotImplementedError()


def aten_fake_quantize_per_channel_affine_cachemask(
    self: TensorType,
    scale: TensorType,
    zero_point: TensorType,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> tuple[TensorType, TensorType]:
    # fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output, Tensor mask)

    raise NotImplementedError()


def aten_fake_quantize_per_channel_affine_cachemask_backward(
    grad: TensorType, mask: TensorType
) -> TensorType:
    # fake_quantize_per_channel_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor

    raise NotImplementedError()


def aten_fake_quantize_per_tensor_affine(
    self: TensorType, scale: float, zero_point: int, quant_min: int, quant_max: int
) -> TensorType:
    # fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor

    raise NotImplementedError()


def aten_fake_quantize_per_tensor_affine_cachemask(
    self: TensorType, scale: float, zero_point: int, quant_min: int, quant_max: int
) -> tuple[TensorType, TensorType]:
    # fake_quantize_per_tensor_affine_cachemask(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> (Tensor output, Tensor mask)

    raise NotImplementedError()


def aten_fake_quantize_per_tensor_affine_cachemask_backward(
    grad: TensorType, mask: TensorType
) -> TensorType:
    # fake_quantize_per_tensor_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor

    raise NotImplementedError()


def aten_fbgemm_linear_fp16_weight(
    input: TensorType, packed_weight: TensorType, bias: TensorType
) -> TensorType:
    # fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor

    raise NotImplementedError()


def aten_fbgemm_linear_fp16_weight_fp32_activation(
    input: TensorType, packed_weight: TensorType, bias: TensorType
) -> TensorType:
    # fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor

    raise NotImplementedError()


def aten_fbgemm_linear_int8_weight(
    input: TensorType,
    weight: TensorType,
    packed: TensorType,
    col_offsets: TensorType,
    weight_scale: float,
    weight_zero_point: float,
    bias: TensorType,
) -> TensorType:
    # fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor

    raise NotImplementedError()


def aten_fbgemm_linear_int8_weight_fp32_activation(
    input: TensorType,
    weight: TensorType,
    packed: TensorType,
    col_offsets: TensorType,
    weight_scale: float,
    weight_zero_point: float,
    bias: TensorType,
) -> TensorType:
    # fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor

    raise NotImplementedError()


def aten_fbgemm_linear_quantize_weight(
    input: TensorType,
) -> tuple[TensorType, TensorType, float, int]:
    # fbgemm_linear_quantize_weight(Tensor input) -> (Tensor, Tensor, float, int)

    raise NotImplementedError()


def aten_fbgemm_pack_gemm_matrix_fp16(input: TensorType) -> TensorType:
    # fbgemm_pack_gemm_matrix_fp16(Tensor input) -> Tensor

    raise NotImplementedError()


def aten_fbgemm_pack_quantized_matrix(input: TensorType) -> TensorType:
    # fbgemm_pack_quantized_matrix(Tensor input) -> Tensor

    raise NotImplementedError()


def aten_feature_alpha_dropout(input: TensorType, p: float, train: bool) -> TensorType:
    # feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor

    raise NotImplementedError()


def aten_feature_dropout(input: TensorType, p: float, train: bool) -> TensorType:
    # feature_dropout(Tensor input, float p, bool train) -> Tensor

    raise NotImplementedError()


def aten_fill(self: TensorType, value: TensorType) -> TensorType:
    # fill.Tensor(Tensor self, Tensor value) -> Tensor

    raise NotImplementedError()


def aten_fix(self: TensorType) -> TensorType:
    # fix(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_flip(self: TensorType, dims: Sequence[int]) -> TensorType:
    # flip(Tensor self, int[] dims) -> Tensor

    raise NotImplementedError()


def aten_fliplr(self: TensorType) -> TensorType:
    # fliplr(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_flipud(self: TensorType) -> TensorType:
    # flipud(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_floor(self: TensorType) -> TensorType:
    # floor(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_floor_divide(self: TensorType, other: TensorType) -> TensorType:
    # floor_divide(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_fmax(self: TensorType, other: TensorType) -> TensorType:
    # fmax(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_fmin(self: TensorType, other: TensorType) -> TensorType:
    # fmin(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_fmod(self: TensorType, other: TensorType) -> TensorType:
    # fmod.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_frac(self: TensorType) -> TensorType:
    # frac(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_frexp(self: TensorType) -> tuple[TensorType, TensorType]:
    # frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)

    raise NotImplementedError()


def aten_frobenius_norm(self: TensorType) -> TensorType:
    # frobenius_norm(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_from_file(
    filename: str, shared: Optional[bool] = None, size: Optional[int] = 0
) -> TensorType:
    # from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_full(size: INT64, fill_value: float) -> TensorType:
    # full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_full_like(
    self: TensorType, fill_value: float, memory_format: Optional[str] = None
) -> TensorType:
    # full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    raise NotImplementedError()


def aten_fused_moving_avg_obs_fake_quant(
    self: TensorType,
    observer_on: TensorType,
    fake_quant_on: TensorType,
    running_min: TensorType,
    running_max: TensorType,
    scale: TensorType,
    zero_point: TensorType,
    averaging_const: float,
    quant_min: int,
    quant_max: int,
    ch_axis: int,
    per_row_fake_quant: bool = False,
    symmetric_quant: bool = False,
) -> TensorType:
    # fused_moving_avg_obs_fake_quant(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> Tensor

    raise NotImplementedError()


def aten_gather(
    self: TensorType, dim: int, index: TensorType, sparse_grad: bool = False
) -> TensorType:
    # gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor

    raise NotImplementedError()


def aten_gather_backward(
    grad: TensorType, self: TensorType, dim: int, index: TensorType, sparse_grad: bool
) -> TensorType:
    # gather_backward(Tensor grad, Tensor self, int dim, Tensor index, bool sparse_grad) -> Tensor

    raise NotImplementedError()


def aten_gcd(self: TensorType, other: TensorType) -> TensorType:
    # gcd(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_ge(self: TensorType, other: TensorType) -> TensorType:
    # ge.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_geqrf(self: TensorType) -> tuple[TensorType, TensorType]:
    # geqrf(Tensor self) -> (Tensor a, Tensor tau)

    raise NotImplementedError()


def aten_ger(self: TensorType, vec2: TensorType) -> TensorType:
    # ger(Tensor self, Tensor vec2) -> Tensor

    raise NotImplementedError()


def aten_greater(self: TensorType, other: TensorType) -> TensorType:
    # greater.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_greater_equal(self: TensorType, other: TensorType) -> TensorType:
    # greater_equal.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_grid_sampler(
    input: TensorType,
    grid: TensorType,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TensorType:
    # grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor

    raise NotImplementedError()


def aten_grid_sampler_2d(
    input: TensorType,
    grid: TensorType,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TensorType:
    # grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor

    raise NotImplementedError()


def aten_grid_sampler_2d_backward(
    grad_output: TensorType,
    input: TensorType,
    grid: TensorType,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType]:
    # grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_grid_sampler_3d(
    input: TensorType,
    grid: TensorType,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TensorType:
    # grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor

    raise NotImplementedError()


def aten_grid_sampler_3d_backward(
    grad_output: TensorType,
    input: TensorType,
    grid: TensorType,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType]:
    # grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_group_norm(
    input: TensorType,
    num_groups: int,
    weight: Optional[TensorType] = None,
    bias: Optional[TensorType] = None,
    eps: float = 1e-05,
    cudnn_enabled: bool = True,
) -> TensorType:
    # group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor

    raise NotImplementedError()


def aten_gru_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> TensorType:
    # gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor

    raise NotImplementedError()


def aten_gt(self, other):
    # gt.Tensor(Tensor self, Tensor other) -> Tensor

    # TODO(justinchuby): Input spec: non bool tensor
    # Boolean inputs can be pre-casted by policy
    return op.Greater(self, other)


def aten_hamming_window(window_length: int) -> TensorType:
    # hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_hann_window(window_length: int) -> TensorType:
    # hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_hardshrink(self: TensorType, lambd: float = 0.5) -> TensorType:
    # hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor

    raise NotImplementedError()


def aten_hardshrink_backward(
    grad_out: TensorType, self: TensorType, lambd: float
) -> TensorType:
    # hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor

    raise NotImplementedError()


def aten_heaviside(self: TensorType, values: TensorType) -> TensorType:
    # heaviside(Tensor self, Tensor values) -> Tensor

    raise NotImplementedError()


def aten_hinge_embedding_loss(
    self: TensorType, target: TensorType, margin: float = 1.0, reduction: int = 1
) -> TensorType:
    # hinge_embedding_loss(Tensor self, Tensor target, float margin=1.0, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_histc(
    self: TensorType, bins: int = 100, min: float = 0, max: float = 0
) -> TensorType:
    # histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor

    raise NotImplementedError()


def aten_histogramdd(
    self: TensorType,
    bins: Sequence[int],
    range: Optional[float] = None,
    weight: Optional[TensorType] = None,
    density: bool = False,
) -> tuple[TensorType, TensorType]:
    # histogramdd(Tensor self, int[] bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)

    raise NotImplementedError()


def aten_hspmm(mat1: TensorType, mat2: TensorType) -> TensorType:
    # hspmm(Tensor mat1, Tensor mat2) -> Tensor

    raise NotImplementedError()


def aten_hstack(tensors: Sequence[TensorType]) -> TensorType:
    # hstack(Tensor[] tensors) -> Tensor

    raise NotImplementedError()


def aten_hypot(self: TensorType, other: TensorType) -> TensorType:
    # hypot(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_i0(self: TensorType) -> TensorType:
    # i0(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_igamma(self: TensorType, other: TensorType) -> TensorType:
    # igamma(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_igammac(self: TensorType, other: TensorType) -> TensorType:
    # igammac(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_imag(self: TensorType) -> TensorType:
    # imag(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_index(self: TensorType, indices: Optional[Sequence[TensorType]]) -> TensorType:
    # index.Tensor(Tensor self, Tensor?[] indices) -> Tensor

    raise NotImplementedError()


def aten_index_add(
    self: TensorType, dim: int, index: TensorType, source: TensorType, alpha: float = 1
) -> TensorType:
    # index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor

    raise NotImplementedError()


def aten_index_copy(
    self: TensorType, dim: int, index: TensorType, source: TensorType
) -> TensorType:
    # index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor

    raise NotImplementedError()


def aten_index_put(
    self: TensorType,
    indices: Optional[Sequence[TensorType]],
    values: TensorType,
    accumulate: bool = False,
) -> TensorType:
    # index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor

    raise NotImplementedError()


def aten_index_reduce(
    self: TensorType,
    dim: int,
    index: TensorType,
    source: TensorType,
    reduce: str,
    include_self: bool = True,
) -> TensorType:
    # index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor

    raise NotImplementedError()


def aten_index_select(self: TensorType, dim: int, index: TensorType) -> TensorType:
    # index_select(Tensor self, int dim, Tensor index) -> Tensor

    raise NotImplementedError()


def aten_index_select_backward(
    grad: TensorType, self_sizes: INT64, dim: int, index: TensorType
) -> TensorType:
    # index_select_backward(Tensor grad, SymInt[] self_sizes, int dim, Tensor index) -> Tensor

    raise NotImplementedError()


def aten_indices(self: TensorType) -> TensorType:
    # indices(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_indices_copy(self: TensorType) -> TensorType:
    # indices_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_inner(self: TensorType, other: TensorType) -> TensorType:
    # inner(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_instance_norm(
    input: TensorType,
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    use_input_stats: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool,
) -> TensorType:
    # instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor

    raise NotImplementedError()


def aten_int_repr(self: TensorType) -> TensorType:
    # int_repr(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_inverse(self: TensorType) -> TensorType:
    # inverse(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_is_coalesced(self: TensorType) -> bool:
    # is_coalesced(Tensor self) -> bool

    raise NotImplementedError()


def aten_is_complex(self: TensorType) -> bool:
    # is_complex(Tensor self) -> bool

    raise NotImplementedError()


def aten_is_conj(self: TensorType) -> bool:
    # is_conj(Tensor self) -> bool

    raise NotImplementedError()


def aten_is_distributed(self: TensorType) -> bool:
    # is_distributed(Tensor self) -> bool

    raise NotImplementedError()


def aten_is_floating_point(self: TensorType) -> bool:
    # is_floating_point(Tensor self) -> bool

    raise NotImplementedError()


def aten_is_inference(self: TensorType) -> bool:
    # is_inference(Tensor self) -> bool

    raise NotImplementedError()


def aten_is_leaf(self: TensorType) -> bool:
    # is_leaf(Tensor self) -> bool

    raise NotImplementedError()


def aten_is_neg(self: TensorType) -> bool:
    # is_neg(Tensor self) -> bool

    raise NotImplementedError()


def aten_is_nonzero(self: TensorType) -> bool:
    # is_nonzero(Tensor self) -> bool

    raise NotImplementedError()


def aten_is_pinned(self: TensorType, device: Optional[str] = None) -> bool:
    # is_pinned(Tensor self, Device? device=None) -> bool

    raise NotImplementedError()


def aten_is_same_size(self: TensorType, other: TensorType) -> bool:
    # is_same_size(Tensor self, Tensor other) -> bool

    raise NotImplementedError()


def aten_is_set_to(self: TensorType, tensor: TensorType) -> bool:
    # is_set_to(Tensor self, Tensor tensor) -> bool

    raise NotImplementedError()


def aten_is_signed(self: TensorType) -> bool:
    # is_signed(Tensor self) -> bool

    raise NotImplementedError()


def aten_is_vulkan_available() -> bool:
    # is_vulkan_available() -> bool

    raise NotImplementedError()


def aten_isclose(
    self: TensorType,
    other: TensorType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> TensorType:
    # isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor

    raise NotImplementedError()


def aten_isfinite(self: TensorType) -> TensorType:
    # isfinite(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_isinf(self: TensorType) -> TensorType:
    # isinf(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_isnan(self: TensorType) -> TensorType:
    # isnan(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_isneginf(self: TensorType) -> TensorType:
    # isneginf(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_isposinf(self: TensorType) -> TensorType:
    # isposinf(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_isreal(self: TensorType) -> TensorType:
    # isreal(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_istft(
    self: TensorType,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[TensorType] = None,
    center: bool = True,
    normalized: bool = False,
    onesided: Optional[bool] = None,
    length: Optional[int] = None,
    return_complex: bool = False,
) -> TensorType:
    # istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False) -> Tensor

    raise NotImplementedError()


def aten_item(self: TensorType) -> float:
    # item(Tensor self) -> Scalar

    raise NotImplementedError()


def aten_kaiser_window(window_length: int) -> TensorType:
    # kaiser_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_kl_div(
    self: TensorType, target: TensorType, reduction: int = 1, log_target: bool = False
) -> TensorType:
    # kl_div(Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor

    raise NotImplementedError()


def aten_kron(self: TensorType, other: TensorType) -> TensorType:
    # kron(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_kthvalue(
    self: TensorType, k: int, dim: int = -1, keepdim: bool = False
) -> tuple[TensorType, TensorType]:
    # kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)

    raise NotImplementedError()


def aten_layer_norm(
    input: TensorType,
    normalized_shape: Sequence[int],
    weight: Optional[TensorType] = None,
    bias: Optional[TensorType] = None,
    eps: float = 1e-05,
    cudnn_enable: bool = True,
) -> TensorType:
    # layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor

    raise NotImplementedError()


def aten_lcm(self: TensorType, other: TensorType) -> TensorType:
    # lcm(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_ldexp(self: TensorType, other: TensorType) -> TensorType:
    # ldexp.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_le(self: TensorType, other: TensorType) -> TensorType:
    # le.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_lerp(self: TensorType, end: TensorType, weight: TensorType) -> TensorType:
    # lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor

    raise NotImplementedError()


def aten_less(self: TensorType, other: TensorType) -> TensorType:
    # less.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_less_equal(self: TensorType, other: TensorType) -> TensorType:
    # less_equal.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_lgamma(self: TensorType) -> TensorType:
    # lgamma(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_lift(self: TensorType) -> TensorType:
    # lift(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_lift_fresh(self: TensorType) -> TensorType:
    # lift_fresh(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_lift_fresh_copy(self: TensorType) -> TensorType:
    # lift_fresh_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_linear_backward(
    self: TensorType, grad_output: TensorType, weight: TensorType, output_mask: Sequence[bool]
) -> tuple[TensorType, TensorType, TensorType]:
    # linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_linspace(start: float, end: float, steps: int) -> TensorType:
    # linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_log(self: TensorType) -> TensorType:
    # log(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_log10(self: TensorType) -> TensorType:
    # log10(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_log1p(self: TensorType) -> TensorType:
    # log1p(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_log2(self: TensorType) -> TensorType:
    # log2(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_logaddexp(self: TensorType, other: TensorType) -> TensorType:
    # logaddexp(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_logaddexp2(self: TensorType, other: TensorType) -> TensorType:
    # logaddexp2(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_logcumsumexp(self: TensorType, dim: int) -> TensorType:
    # logcumsumexp(Tensor self, int dim) -> Tensor

    raise NotImplementedError()


def aten_logdet(self: TensorType) -> TensorType:
    # logdet(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_logical_and(self: TensorType, other: TensorType) -> TensorType:
    # logical_and(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_logical_not(self: TensorType) -> TensorType:
    # logical_not(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_logical_or(self: TensorType, other: TensorType) -> TensorType:
    # logical_or(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_logical_xor(self: TensorType, other: TensorType) -> TensorType:
    # logical_xor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_logit(self: TensorType, eps: Optional[float] = None) -> TensorType:
    # logit(Tensor self, float? eps=None) -> Tensor

    raise NotImplementedError()


def aten_logspace(start: float, end: float, steps: int, base: float = 10.0) -> TensorType:
    # logspace(Scalar start, Scalar end, int steps, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_logsumexp(self: TensorType, dim: Sequence[int], keepdim: bool = False) -> TensorType:
    # logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor

    raise NotImplementedError()


def aten_lshift(self: TensorType, other: TensorType) -> TensorType:
    # __lshift__.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_lstm_cell(
    input: TensorType,
    hx: Sequence[TensorType],
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> tuple[TensorType, TensorType]:
    # lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_lstm_mps_backward(
    grad_y: TensorType,
    grad_hy: Optional[TensorType],
    grad_cy: Optional[TensorType],
    z_state: TensorType,
    cell_state_fwd: TensorType,
    input: TensorType,
    hx: Sequence[TensorType],
    params: Sequence[TensorType],
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
) -> tuple[TensorType, TensorType, TensorType]:
    # lstm_mps_backward(Tensor grad_y, Tensor? grad_hy, Tensor? grad_cy, Tensor z_state, Tensor cell_state_fwd, Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor[], Tensor[])

    raise NotImplementedError()


def aten_lt(self, other):
    # lt.Tensor(Tensor self, Tensor other) -> Tensor

    # TODO(justinchuby): Input spec: non bool tensor
    # Boolean inputs can be pre-casted by policy
    return op.Less(self, other)


def aten_lu_solve(self: TensorType, LU_data: TensorType, LU_pivots: TensorType) -> TensorType:
    # lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor

    raise NotImplementedError()


def aten_lu_unpack(
    LU_data: TensorType,
    LU_pivots: TensorType,
    unpack_data: bool = True,
    unpack_pivots: bool = True,
) -> tuple[TensorType, TensorType, TensorType]:
    # lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)

    raise NotImplementedError()


def aten_mH(self: TensorType) -> TensorType:
    # mH(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_mT(self: TensorType) -> TensorType:
    # mT(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_margin_ranking_loss(
    input1: TensorType,
    input2: TensorType,
    target: TensorType,
    margin: float = 0.0,
    reduction: int = 1,
) -> TensorType:
    # margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_masked_fill(self: TensorType, mask: TensorType, value: TensorType) -> TensorType:
    # masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor

    raise NotImplementedError()


def aten_masked_scatter(self: TensorType, mask: TensorType, source: TensorType) -> TensorType:
    # masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor

    raise NotImplementedError()


def aten_masked_select(self: TensorType, mask: TensorType) -> TensorType:
    # masked_select(Tensor self, Tensor mask) -> Tensor

    raise NotImplementedError()


def aten_masked_select_backward(
    grad: TensorType, input: TensorType, mask: TensorType
) -> TensorType:
    # masked_select_backward(Tensor grad, Tensor input, Tensor mask) -> Tensor

    raise NotImplementedError()


def aten_matmul(self, other):
    # matmul(Tensor self, Tensor other) -> Tensor

    return op.MatMul(self, other)


def aten_matmul_backward(
    grad: TensorType, self: TensorType, other: TensorType, mask: Sequence[bool]
) -> tuple[TensorType, TensorType]:
    # matmul_backward(Tensor grad, Tensor self, Tensor other, bool[2] mask) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_matrix_H(self: TensorType) -> TensorType:
    # matrix_H(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_matrix_exp(self: TensorType) -> TensorType:
    # matrix_exp(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_matrix_exp_backward(self: TensorType, grad: TensorType) -> TensorType:
    # matrix_exp_backward(Tensor self, Tensor grad) -> Tensor

    raise NotImplementedError()


def aten_matrix_power(self: TensorType, n: int) -> TensorType:
    # matrix_power(Tensor self, int n) -> Tensor

    raise NotImplementedError()


def aten_max(self: TensorType) -> TensorType:
    # max(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_max_pool1d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    ceil_mode: bool = False,
) -> TensorType:
    # max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor

    raise NotImplementedError()


def aten_max_pool1d_with_indices(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    ceil_mode: bool = False,
) -> tuple[TensorType, TensorType]:
    # max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_max_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> TensorType:
    # max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor

    raise NotImplementedError()


def aten_max_pool3d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    ceil_mode: bool = False,
) -> TensorType:
    # max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor

    raise NotImplementedError()


def aten_maximum(self: TensorType, other: TensorType) -> TensorType:
    # maximum(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_mean(self: TensorType, dtype: Optional[int] = None) -> TensorType:
    # mean(Tensor self, *, ScalarType? dtype=None) -> Tensor

    raise NotImplementedError()


def aten_median(self: TensorType) -> TensorType:
    # median(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_meshgrid(tensors: Sequence[TensorType]) -> TensorType:
    # meshgrid(Tensor[] tensors) -> Tensor[]

    raise NotImplementedError()


def aten_min(self: TensorType) -> TensorType:
    # min(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_minimum(self: TensorType, other: TensorType) -> TensorType:
    # minimum(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_miopen_batch_norm(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
) -> tuple[TensorType, TensorType, TensorType]:
    # miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_miopen_batch_norm_backward(
    input: TensorType,
    grad_output: TensorType,
    weight: TensorType,
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    save_mean: Optional[TensorType],
    save_var: Optional[TensorType],
    epsilon: float,
) -> tuple[TensorType, TensorType, TensorType]:
    # miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_miopen_convolution(
    self: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    padding: INT64,
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    benchmark: bool,
    deterministic: bool,
) -> TensorType:
    # miopen_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor

    raise NotImplementedError()


def aten_miopen_convolution_add_relu(
    self: TensorType,
    weight: TensorType,
    z: TensorType,
    alpha: Optional[float],
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> TensorType:
    # miopen_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor

    raise NotImplementedError()


def aten_miopen_convolution_relu(
    self: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> TensorType:
    # miopen_convolution_relu(Tensor self, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor

    raise NotImplementedError()


def aten_miopen_convolution_transpose(
    self: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    padding: INT64,
    output_padding: INT64,
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    benchmark: bool,
    deterministic: bool,
) -> TensorType:
    # miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor

    raise NotImplementedError()


def aten_miopen_depthwise_convolution(
    self: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    padding: INT64,
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    benchmark: bool,
    deterministic: bool,
) -> TensorType:
    # miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor

    raise NotImplementedError()


def aten_miopen_rnn(
    input: TensorType,
    weight: Sequence[TensorType],
    weight_stride0: int,
    hx: TensorType,
    cx: Optional[TensorType],
    mode: int,
    hidden_size: int,
    num_layers: int,
    batch_first: bool,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_sizes: Sequence[int],
    dropout_state: Optional[TensorType],
) -> tuple[TensorType, TensorType, TensorType, TensorType, TensorType]:
    # miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_miopen_rnn_backward(
    input: TensorType,
    weight: Sequence[TensorType],
    weight_stride0: int,
    weight_buf: TensorType,
    hx: TensorType,
    cx: Optional[TensorType],
    output: TensorType,
    grad_output: Optional[TensorType],
    grad_hy: Optional[TensorType],
    grad_cy: Optional[TensorType],
    mode: int,
    hidden_size: int,
    num_layers: int,
    batch_first: bool,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_sizes: Sequence[int],
    dropout_state: Optional[TensorType],
    reserve: TensorType,
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType, TensorType, TensorType]:
    # miopen_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])

    raise NotImplementedError()


def aten_mkldnn_adaptive_avg_pool2d(
    self: TensorType, output_size: Sequence[int]
) -> TensorType:
    # mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor

    raise NotImplementedError()


def aten_mkldnn_adaptive_avg_pool2d_backward(
    grad_output: TensorType, self: TensorType
) -> TensorType:
    # mkldnn_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor

    raise NotImplementedError()


def aten_mkldnn_convolution(
    self: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    padding: INT64,
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> TensorType:
    # mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, int[] stride, int[] dilation, int groups) -> Tensor

    raise NotImplementedError()


def aten_mkldnn_linear_backward(
    self: TensorType, grad_output: TensorType, weight: TensorType, output_mask: Sequence[bool]
) -> tuple[TensorType, TensorType, TensorType]:
    # mkldnn_linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_mkldnn_linear_backward_input(
    input_size: Sequence[int], grad_output: TensorType, weight: TensorType
) -> TensorType:
    # mkldnn_linear_backward_input(int[] input_size, Tensor grad_output, Tensor weight) -> Tensor

    raise NotImplementedError()


def aten_mkldnn_linear_backward_weights(
    grad_output: TensorType, input: TensorType, weight: TensorType, bias_defined: bool
) -> tuple[TensorType, TensorType]:
    # mkldnn_linear_backward_weights(Tensor grad_output, Tensor input, Tensor weight, bool bias_defined) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_mkldnn_max_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> TensorType:
    # mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor

    raise NotImplementedError()


def aten_mkldnn_max_pool2d_backward(
    grad_output: TensorType,
    output: TensorType,
    input: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> TensorType:
    # mkldnn_max_pool2d_backward(Tensor grad_output, Tensor output, Tensor input, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor

    raise NotImplementedError()


def aten_mkldnn_max_pool3d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    ceil_mode: bool = False,
) -> TensorType:
    # mkldnn_max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor

    raise NotImplementedError()


def aten_mkldnn_max_pool3d_backward(
    grad_output: TensorType,
    output: TensorType,
    input: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    ceil_mode: bool = False,
) -> TensorType:
    # mkldnn_max_pool3d_backward(Tensor grad_output, Tensor output, Tensor input, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor

    raise NotImplementedError()


def aten_mm(self, mat2):
    # mm(Tensor self, Tensor mat2) -> Tensor

    # TODO(justinchuby): Specify type conversion for uint8/int8/int16
    return op.MatMul(self, mat2)


def aten_mode(
    self: TensorType, dim: int = -1, keepdim: bool = False
) -> tuple[TensorType, TensorType]:
    # mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)

    raise NotImplementedError()


def aten_mps_convolution_backward(
    self: TensorType,
    grad_output: TensorType,
    weight: TensorType,
    padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType, TensorType]:
    # mps_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_mps_convolution_transpose_backward(
    self: TensorType,
    grad_output: TensorType,
    weight: TensorType,
    padding: Sequence[int],
    output_padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType]:
    # mps_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool[2] output_mask) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_mps_max_pool2d_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> TensorType:
    # mps_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor

    raise NotImplementedError()


def aten_msort(self: TensorType) -> TensorType:
    # msort(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_mul(self, other) -> TensorType:
    # mul.Tensor(Tensor self, Tensor other) -> Tensor

    return op.Mul(self, other)


def aten_mul_bool(self: BOOL, other: BOOL) -> BOOL:
    """ONNX Mul doesn't support Boolean, so use And as an equivalent operator."""

    # TODO(justinchuby): Handle cases where type reconcilation is not enough,
    # since different ONNX operators are used based on different data types.

    return op.And(self, other)


def aten_multinomial(
    self: TensorType,
    num_samples: int,
    replacement: bool = False,
    generator: Optional[str] = None,
) -> TensorType:
    # multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor

    raise NotImplementedError()


def aten_multiply(self: TensorType, other: TensorType) -> TensorType:
    # multiply.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_mv(self: TensorType, vec: TensorType) -> TensorType:
    # mv(Tensor self, Tensor vec) -> Tensor

    raise NotImplementedError()


def aten_mvlgamma(self: TensorType, p: int) -> TensorType:
    # mvlgamma(Tensor self, int p) -> Tensor

    raise NotImplementedError()


def aten_nan_to_num(
    self: TensorType,
    nan: Optional[float] = None,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
) -> TensorType:
    # nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor

    raise NotImplementedError()


def aten_nanmean(
    self: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    dtype: Optional[int] = None,
) -> TensorType:
    # nanmean(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor

    raise NotImplementedError()


def aten_nanmedian(self: TensorType) -> TensorType:
    # nanmedian(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_nanquantile(
    self: TensorType,
    q: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    interpolation: str = "linear",
) -> TensorType:
    # nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor

    raise NotImplementedError()


def aten_nansum(
    self: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    dtype: Optional[int] = None,
) -> TensorType:
    # nansum(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor

    raise NotImplementedError()


def aten_narrow(self: TensorType, dim: int, start: INT64, length: INT64) -> TensorType:
    # narrow(Tensor(a) self, int dim, SymInt start, SymInt length) -> Tensor(a)

    raise NotImplementedError()


def aten_narrow_copy(self: TensorType, dim: int, start: INT64, length: INT64) -> TensorType:
    # narrow_copy(Tensor self, int dim, SymInt start, SymInt length) -> Tensor

    raise NotImplementedError()


def aten_native_batch_norm(
    input: TensorType,
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[TensorType, TensorType, TensorType]:
    # native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_native_batch_norm_backward(
    grad_out: TensorType,
    input: TensorType,
    weight: Optional[TensorType],
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    save_mean: Optional[TensorType],
    save_invstd: Optional[TensorType],
    train: bool,
    eps: float,
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType, TensorType]:
    # native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_native_channel_shuffle(self: TensorType, groups: int) -> TensorType:
    # native_channel_shuffle(Tensor self, int groups) -> Tensor

    raise NotImplementedError()


def aten_native_dropout(
    input: TensorType, p: float, train: Optional[bool]
) -> tuple[TensorType, TensorType]:
    # native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_native_dropout_backward(
    grad_output: TensorType, mask: TensorType, scale: float
) -> TensorType:
    # native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor

    raise NotImplementedError()


def aten_native_group_norm(
    input: TensorType,
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    N: INT64,
    C: INT64,
    HxW: INT64,
    group: int,
    eps: float,
) -> tuple[TensorType, TensorType, TensorType]:
    # native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_native_group_norm_backward(
    grad_out: TensorType,
    input: TensorType,
    mean: TensorType,
    rstd: TensorType,
    weight: Optional[TensorType],
    N: INT64,
    C: INT64,
    HxW: INT64,
    group: int,
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType, TensorType]:
    # native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_native_layer_norm(
    input: TensorType,
    normalized_shape: INT64,
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    eps: float,
) -> tuple[TensorType, TensorType, TensorType]:
    # native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_native_layer_norm_backward(
    grad_out: TensorType,
    input: TensorType,
    normalized_shape: INT64,
    mean: TensorType,
    rstd: TensorType,
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType, TensorType]:
    # native_layer_norm_backward(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_native_norm(self: TensorType, p: float = 2) -> TensorType:
    # native_norm(Tensor self, Scalar p=2) -> Tensor

    raise NotImplementedError()


def aten_ne(self: TensorType, other: TensorType) -> TensorType:
    # ne.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_neg(self: TensorType) -> TensorType:
    # neg(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_negative(self: TensorType) -> TensorType:
    # negative(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_new_empty(self: TensorType, size: INT64) -> TensorType:
    # new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_new_empty_strided(self: TensorType, size: INT64, stride: INT64) -> TensorType:
    # new_empty_strided(Tensor self, SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_new_full(self: TensorType, size: INT64, fill_value: float) -> TensorType:
    # new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_new_ones(self: TensorType, size: INT64) -> TensorType:
    # new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_new_zeros(self: TensorType, size: INT64) -> TensorType:
    # new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_nextafter(self: TensorType, other: TensorType) -> TensorType:
    # nextafter(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_nonzero(self: TensorType) -> TensorType:
    # nonzero(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_nonzero_numpy(self: TensorType) -> TensorType:
    # nonzero_numpy(Tensor self) -> Tensor[]

    raise NotImplementedError()


def aten_norm_except_dim(v: TensorType, pow: int = 2, dim: int = 0) -> TensorType:
    # norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor

    raise NotImplementedError()


def aten_normal(
    self: TensorType, mean: float = 0, std: float = 1, generator: Optional[str] = None
) -> TensorType:
    # normal_functional(Tensor self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor

    raise NotImplementedError()


def aten_not_equal(self: TensorType, other: TensorType) -> TensorType:
    # not_equal.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_nuclear_norm(self: TensorType, keepdim: bool = False) -> TensorType:
    # nuclear_norm(Tensor self, bool keepdim=False) -> Tensor

    raise NotImplementedError()


def aten_ones(size: INT64, dtype: int = -1) -> TensorType:
    # ones(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    one = op.Constant(value_float=1)
    if dtype != -1:
        one = op.Cast(one, to=dtype)  # type: ignore[arg-type]
    return op.Expand(one, size)  # type: ignore[arg-type]


def aten_ones_like(self, dtype: int = -1):
    """ones_like.

    Note: dtype is an onnx enum. Users should convert torch dtype to onnx dtype
    before calling this function.
    """
    # ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    shape = op.Shape(self)
    if dtype == -1:
        one = op.CastLike(1, self)  # type: ignore[arg-type]
    else:
        one = op.Cast(1, to=dtype)  # type: ignore[arg-type]
    return op.Expand(one, shape)


def aten_or(self: TensorType, other: TensorType) -> TensorType:
    # __or__.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_orgqr(self: TensorType, input2: TensorType) -> TensorType:
    # orgqr(Tensor self, Tensor input2) -> Tensor

    raise NotImplementedError()


def aten_ormqr(
    self: TensorType,
    input2: TensorType,
    input3: TensorType,
    left: bool = True,
    transpose: bool = False,
) -> TensorType:
    # ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor

    raise NotImplementedError()


def aten_outer(self: TensorType, vec2: TensorType) -> TensorType:
    # outer(Tensor self, Tensor vec2) -> Tensor

    raise NotImplementedError()


def aten_output_nr(self: TensorType) -> int:
    # output_nr(Tensor self) -> int

    raise NotImplementedError()


def aten_pairwise_distance(
    x1: TensorType, x2: TensorType, p: float = 2, eps: float = 1e-06, keepdim: bool = False
) -> TensorType:
    # pairwise_distance(Tensor x1, Tensor x2, float p=2, float eps=1e-06, bool keepdim=False) -> Tensor

    raise NotImplementedError()


def aten_pdist(self: TensorType, p: float = 2) -> TensorType:
    # pdist(Tensor self, float p=2) -> Tensor

    raise NotImplementedError()


def aten_permute(self: TensorType, dims: Sequence[int]) -> TensorType:
    # permute(Tensor(a) self, int[] dims) -> Tensor(a)

    raise NotImplementedError()


def aten_permute_copy(self: TensorType, dims: Sequence[int]) -> TensorType:
    # permute_copy(Tensor self, int[] dims) -> Tensor

    raise NotImplementedError()


def aten_pin_memory(self: TensorType, device: Optional[str] = None) -> TensorType:
    # pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)

    raise NotImplementedError()


def aten_pinverse(self: TensorType, rcond: float = 1e-15) -> TensorType:
    # pinverse(Tensor self, float rcond=1e-15) -> Tensor

    raise NotImplementedError()


def aten_pixel_shuffle(self: TensorType, upscale_factor: int) -> TensorType:
    # pixel_shuffle(Tensor self, int upscale_factor) -> Tensor

    raise NotImplementedError()


def aten_pixel_unshuffle(self: TensorType, downscale_factor: int) -> TensorType:
    # pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor

    raise NotImplementedError()


def aten_poisson(self: TensorType, generator: Optional[str] = None) -> TensorType:
    # poisson(Tensor self, Generator? generator=None) -> Tensor

    raise NotImplementedError()


def aten_poisson_nll_loss(
    input: TensorType,
    target: TensorType,
    log_input: bool,
    full: bool,
    eps: float,
    reduction: int,
) -> TensorType:
    # poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor

    raise NotImplementedError()


def aten_polar(abs: TensorType, angle: TensorType) -> TensorType:
    # polar(Tensor abs, Tensor angle) -> Tensor

    raise NotImplementedError()


def aten_polygamma(n: int, self: TensorType) -> TensorType:
    # polygamma(int n, Tensor self) -> Tensor

    raise NotImplementedError()


def aten_positive(self: TensorType) -> TensorType:
    # positive(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_prelu(self: TensorType, weight: TensorType) -> TensorType:
    # prelu(Tensor self, Tensor weight) -> Tensor

    raise NotImplementedError()


def aten_prelu_backward(
    grad_output: TensorType, self: TensorType, weight: TensorType
) -> tuple[TensorType, TensorType]:
    # prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_prod(self: TensorType, dtype: Optional[int] = None) -> TensorType:
    # prod(Tensor self, *, ScalarType? dtype=None) -> Tensor

    raise NotImplementedError()


def aten_promote_types(type1: int, type2: int) -> int:
    # promote_types(ScalarType type1, ScalarType type2) -> ScalarType

    raise NotImplementedError()


def aten_put(
    self: TensorType, index: TensorType, source: TensorType, accumulate: bool = False
) -> TensorType:
    # put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor

    raise NotImplementedError()


def aten_q_per_channel_axis(self: TensorType) -> int:
    # q_per_channel_axis(Tensor self) -> int

    raise NotImplementedError()


def aten_q_per_channel_scales(self: TensorType) -> TensorType:
    # q_per_channel_scales(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_q_per_channel_zero_points(self: TensorType) -> TensorType:
    # q_per_channel_zero_points(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_q_scale(self: TensorType) -> float:
    # q_scale(Tensor self) -> float

    raise NotImplementedError()


def aten_q_zero_point(self: TensorType) -> int:
    # q_zero_point(Tensor self) -> int

    raise NotImplementedError()


def aten_qr(self: TensorType, some: bool = True) -> tuple[TensorType, TensorType]:
    # qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)

    raise NotImplementedError()


def aten_qscheme(self: TensorType) -> str:
    # qscheme(Tensor self) -> QScheme

    raise NotImplementedError()


def aten_quantile(
    self: TensorType,
    q: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    interpolation: str = "linear",
) -> TensorType:
    # quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor

    raise NotImplementedError()


def aten_quantize_per_channel(
    self: TensorType, scales: TensorType, zero_points: TensorType, axis: int, dtype: int
) -> TensorType:
    # quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) -> Tensor

    raise NotImplementedError()


def aten_quantize_per_tensor(
    self: TensorType, scale: float, zero_point: int, dtype: int
) -> TensorType:
    # quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor

    raise NotImplementedError()


def aten_quantize_per_tensor_dynamic(
    self: TensorType, dtype: int, reduce_range: bool
) -> TensorType:
    # quantize_per_tensor_dynamic(Tensor self, ScalarType dtype, bool reduce_range) -> Tensor

    raise NotImplementedError()


def aten_quantized_batch_norm(
    input: TensorType,
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    mean: TensorType,
    var: TensorType,
    eps: float,
    output_scale: float,
    output_zero_point: int,
) -> TensorType:
    # quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor

    raise NotImplementedError()


def aten_quantized_gru_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: TensorType,
    b_hh: TensorType,
    packed_ih: TensorType,
    packed_hh: TensorType,
    col_offsets_ih: TensorType,
    col_offsets_hh: TensorType,
    scale_ih: float,
    scale_hh: float,
    zero_point_ih: float,
    zero_point_hh: float,
) -> TensorType:
    # quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor

    raise NotImplementedError()


def aten_quantized_lstm_cell(
    input: TensorType,
    hx: Sequence[TensorType],
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: TensorType,
    b_hh: TensorType,
    packed_ih: TensorType,
    packed_hh: TensorType,
    col_offsets_ih: TensorType,
    col_offsets_hh: TensorType,
    scale_ih: float,
    scale_hh: float,
    zero_point_ih: float,
    zero_point_hh: float,
) -> tuple[TensorType, TensorType]:
    # quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_quantized_max_pool1d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    ceil_mode: bool = False,
) -> TensorType:
    # quantized_max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor

    raise NotImplementedError()


def aten_quantized_max_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> TensorType:
    # quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor

    raise NotImplementedError()


def aten_quantized_rnn_relu_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: TensorType,
    b_hh: TensorType,
    packed_ih: TensorType,
    packed_hh: TensorType,
    col_offsets_ih: TensorType,
    col_offsets_hh: TensorType,
    scale_ih: float,
    scale_hh: float,
    zero_point_ih: float,
    zero_point_hh: float,
) -> TensorType:
    # quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor

    raise NotImplementedError()


def aten_quantized_rnn_tanh_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: TensorType,
    b_hh: TensorType,
    packed_ih: TensorType,
    packed_hh: TensorType,
    col_offsets_ih: TensorType,
    col_offsets_hh: TensorType,
    scale_ih: float,
    scale_hh: float,
    zero_point_ih: float,
    zero_point_hh: float,
) -> TensorType:
    # quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor

    raise NotImplementedError()


def aten_rad2deg(self: TensorType) -> TensorType:
    # rad2deg(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_rand(size: INT64) -> TensorType:
    # rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_rand_like(self: TensorType, memory_format: Optional[str] = None) -> TensorType:
    # rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    raise NotImplementedError()


def aten_randint(high: int, size: INT64) -> TensorType:
    # randint(int high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_randint_like(
    self: TensorType, high: int, memory_format: Optional[str] = None
) -> TensorType:
    # randint_like(Tensor self, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    raise NotImplementedError()


def aten_randn(size: INT64) -> TensorType:
    # randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_randn_like(self: TensorType, memory_format: Optional[str] = None) -> TensorType:
    # randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    raise NotImplementedError()


def aten_randperm(n: int) -> TensorType:
    # randperm(int n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_range(start: float, end: float) -> TensorType:
    # range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_ravel(self: TensorType) -> TensorType:
    # ravel(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_real(self: TensorType) -> TensorType:
    # real(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_reciprocal(self: TensorType) -> TensorType:
    # reciprocal(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_record_stream(self: TensorType, s: str) -> Any:
    # record_stream(Tensor(a!) self, Stream s) -> ()

    raise NotImplementedError()


def aten_refine_names(self: TensorType, names: Sequence[str]) -> TensorType:
    # refine_names(Tensor(a) self, Dimname[] names) -> Tensor(a)

    raise NotImplementedError()


def aten_relu(self: TensorType) -> TensorType:
    # relu(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_remainder(self: TensorType, other: TensorType) -> TensorType:
    # remainder.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_rename(self: TensorType, names: Optional[str]) -> TensorType:
    # rename(Tensor(a) self, Dimname[]? names) -> Tensor(a)

    raise NotImplementedError()


def aten_renorm(self: TensorType, p: float, dim: int, maxnorm: float) -> TensorType:
    # renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor

    raise NotImplementedError()


def aten_repeat(self, repeats: INT64):
    # repeat(Tensor self, SymInt[] repeats) -> Tensor

    # FIXME(justinchuby): When repeats.shape == [0]

    # TODO(justinchuby): Make ones_like a function when onnxscript supports it
    # shape = ones_like(repeats) := {
    one = op.Constant(value_int=1)
    repeats_shape = op.Shape(repeats)
    shape = op.Expand(one, repeats_shape)
    # }
    self_expanded = op.Expand(self, shape)  # type: ignore[arg-type]
    return op.Tile(self_expanded, repeats)


def aten_repeat_interleave(
    repeats: TensorType, output_size: Optional[int] = None
) -> TensorType:
    # repeat_interleave.Tensor(Tensor repeats, *, int? output_size=None) -> Tensor

    raise NotImplementedError()


def aten_reshape(self: TensorType, shape: INT64) -> TensorType:
    # reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)

    raise NotImplementedError()


def aten_reshape_as(self: TensorType, other: TensorType) -> TensorType:
    # reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)

    raise NotImplementedError()


def aten_resolve_conj(self: TensorType) -> TensorType:
    # resolve_conj(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_resolve_neg(self: TensorType) -> TensorType:
    # resolve_neg(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_result_type(tensor: TensorType, other: TensorType) -> int:
    # result_type.Tensor(Tensor tensor, Tensor other) -> ScalarType

    raise NotImplementedError()


def aten_retain_grad(self: TensorType) -> Any:
    # retain_grad(Tensor(a!) self) -> ()

    raise NotImplementedError()


def aten_retains_grad(self: TensorType) -> bool:
    # retains_grad(Tensor self) -> bool

    raise NotImplementedError()


def aten_rnn_relu_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> TensorType:
    # rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor

    raise NotImplementedError()


def aten_rnn_tanh_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> TensorType:
    # rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor

    raise NotImplementedError()


def aten_roll(
    self: TensorType, shifts: Sequence[int], dims: Optional[Sequence[int]] = None
) -> TensorType:
    # roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor

    raise NotImplementedError()


def aten_rot90(self: TensorType, k: int = 1, dims: Sequence[int] = (0, 1)) -> TensorType:
    # rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor

    raise NotImplementedError()


def aten_round(self):
    # round(Tensor self) -> Tensor

    return op.Round(self)


def aten_row_indices(self: TensorType) -> TensorType:
    # row_indices(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_row_indices_copy(self: TensorType) -> TensorType:
    # row_indices_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_row_stack(tensors: Sequence[TensorType]) -> TensorType:
    # row_stack(Tensor[] tensors) -> Tensor

    raise NotImplementedError()


def aten_rrelu(
    self: TensorType,
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
    training: bool = False,
    generator: Optional[str] = None,
) -> TensorType:
    # rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor

    raise NotImplementedError()


def aten_rshift(self: TensorType, other: TensorType) -> TensorType:
    # __rshift__.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_rsqrt(self: TensorType) -> TensorType:
    # rsqrt(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_rsub(self: TensorType, other: TensorType, alpha: float = 1) -> TensorType:
    # rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

    raise NotImplementedError()


def aten_scalar_tensor(s: float) -> TensorType:
    # scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_scatter_add(
    self: TensorType, dim: int, index: TensorType, src: TensorType
) -> TensorType:
    # scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor

    raise NotImplementedError()


def aten_searchsorted(
    sorted_sequence: TensorType,
    self: TensorType,
    out_int32: bool = False,
    right: bool = False,
    side: Optional[str] = None,
    sorter: Optional[TensorType] = None,
) -> TensorType:
    # searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor

    raise NotImplementedError()


def aten_segment_reduce(
    data: TensorType,
    reduce: str,
    lengths: Optional[TensorType] = None,
    indices: Optional[TensorType] = None,
    offsets: Optional[TensorType] = None,
    axis: int = 0,
    unsafe: bool = False,
    initial: Optional[float] = None,
) -> TensorType:
    # segment_reduce(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, Tensor? offsets=None, int axis=0, bool unsafe=False, Scalar? initial=None) -> Tensor

    raise NotImplementedError()


def aten_select_backward(
    grad_output: TensorType, input_sizes: INT64, dim: int, index: int
) -> TensorType:
    # select_backward(Tensor grad_output, SymInt[] input_sizes, int dim, int index) -> Tensor

    raise NotImplementedError()


def aten_select_scatter(self: TensorType, src: TensorType, dim: int, index: int) -> TensorType:
    # select_scatter(Tensor self, Tensor src, int dim, int index) -> Tensor

    raise NotImplementedError()


def aten_selu(self):
    # selu(Tensor self) -> Tensor

    return op.Selu(self)


def aten_set_data(self: TensorType, new_data: TensorType) -> Any:
    # set_data(Tensor(a!) self, Tensor new_data) -> ()

    raise NotImplementedError()


def aten_sgn(self: TensorType) -> TensorType:
    # sgn(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_sigmoid(self: TensorType) -> TensorType:
    # sigmoid(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_sign(self: TensorType) -> TensorType:
    # sign(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_signbit(self: TensorType) -> TensorType:
    # signbit(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_sin(self: TensorType) -> TensorType:
    # sin(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_sinc(self: TensorType) -> TensorType:
    # sinc(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_sinh(self: TensorType) -> TensorType:
    # sinh(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_slice(
    self: TensorType,
    dim: int = 0,
    start: Optional[INT64] = None,
    end: Optional[INT64] = None,
    step: INT64 = 1,
) -> TensorType:
    # slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)

    raise NotImplementedError()


def aten_slice_backward(
    grad_output: TensorType,
    input_sizes: INT64,
    dim: int,
    start: INT64,
    end: INT64,
    step: INT64,
) -> TensorType:
    # slice_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step) -> Tensor

    raise NotImplementedError()


def aten_slice_copy(
    self: TensorType,
    dim: int = 0,
    start: Optional[INT64] = None,
    end: Optional[INT64] = None,
    step: INT64 = 1,
) -> TensorType:
    # slice_copy.Tensor(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor

    raise NotImplementedError()


def aten_slice_scatter(
    self: TensorType,
    src: TensorType,
    dim: int = 0,
    start: Optional[INT64] = None,
    end: Optional[INT64] = None,
    step: INT64 = 1,
) -> TensorType:
    # slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor

    raise NotImplementedError()


def aten_slogdet(self: TensorType) -> tuple[TensorType, TensorType]:
    # slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)

    raise NotImplementedError()


def aten_smm(self: TensorType, mat2: TensorType) -> TensorType:
    # smm(Tensor self, Tensor mat2) -> Tensor

    raise NotImplementedError()


def aten_sort(
    self: TensorType, dim: int = -1, descending: bool = False
) -> tuple[TensorType, TensorType]:
    # sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)

    raise NotImplementedError()


def aten_sparse_dim(self: TensorType) -> int:
    # sparse_dim(Tensor self) -> int

    raise NotImplementedError()


def aten_sparse_mask(self: TensorType, mask: TensorType) -> TensorType:
    # sparse_mask(Tensor self, Tensor mask) -> Tensor

    raise NotImplementedError()


def aten_split(self: TensorType, split_size: INT64, dim: int = 0) -> TensorType:
    # split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]

    raise NotImplementedError()


def aten_split_copy(self: TensorType, split_size: INT64, dim: int = 0) -> TensorType:
    # split_copy.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]

    raise NotImplementedError()


def aten_split_with_sizes(self: TensorType, split_sizes: INT64, dim: int = 0) -> TensorType:
    # split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]

    raise NotImplementedError()


def aten_split_with_sizes_copy(
    self: TensorType, split_sizes: INT64, dim: int = 0
) -> TensorType:
    # split_with_sizes_copy(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]

    raise NotImplementedError()


def aten_sqrt(self: TensorType) -> TensorType:
    # sqrt(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_square(self: TensorType) -> TensorType:
    # square(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_squeeze(self: TensorType) -> TensorType:
    # squeeze(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_squeeze_copy(self: TensorType) -> TensorType:
    # squeeze_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_sspaddmm(
    self: TensorType, mat1: TensorType, mat2: TensorType, beta: float = 1, alpha: float = 1
) -> TensorType:
    # sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

    raise NotImplementedError()


def aten_stack(tensors: Sequence[TensorType], dim: int = 0) -> TensorType:
    # stack(Tensor[] tensors, int dim=0) -> Tensor

    raise NotImplementedError()


def aten_std(self: TensorType, unbiased: bool = True) -> TensorType:
    # std(Tensor self, bool unbiased=True) -> Tensor

    raise NotImplementedError()


def aten_std_mean(self: TensorType, unbiased: bool = True) -> tuple[TensorType, TensorType]:
    # std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_stft(
    self: TensorType,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[TensorType] = None,
    normalized: bool = False,
    onesided: Optional[bool] = None,
    return_complex: Optional[bool] = None,
) -> TensorType:
    # stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor

    raise NotImplementedError()


def aten_sub(self, other, alpha: float = 1) -> TensorType:
    # sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

    if alpha != 1:
        other = op.Mul(other, alpha)  # type: ignore[arg-type]

    return op.Sub(self, other)


def aten_subtract(self: TensorType, other: TensorType, alpha: float = 1) -> TensorType:
    # subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

    raise NotImplementedError()


def aten_sum(self: TensorType, dtype: Optional[int] = None) -> TensorType:
    # sum(Tensor self, *, ScalarType? dtype=None) -> Tensor

    raise NotImplementedError()


def aten_sum_to_size(self: TensorType, size: Sequence[int]) -> TensorType:
    # sum_to_size(Tensor self, int[] size) -> Tensor

    raise NotImplementedError()


def aten_svd(
    self: TensorType, some: bool = True, compute_uv: bool = True
) -> tuple[TensorType, TensorType, TensorType]:
    # svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)

    raise NotImplementedError()


def aten_swapaxes(self: TensorType, axis0: int, axis1: int) -> TensorType:
    # swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)

    raise NotImplementedError()


def aten_swapdims(self: TensorType, dim0: int, dim1: int) -> TensorType:
    # swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)

    raise NotImplementedError()


def aten_symeig(
    self: TensorType, eigenvectors: bool = False, upper: bool = True
) -> tuple[TensorType, TensorType]:
    # symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)

    raise NotImplementedError()


def aten_t(self: TensorType) -> TensorType:
    # t(Tensor(a) self) -> Tensor(a)

    # TODO(justinchuby): Make rank a function
    rank = op.Size(op.Shape(self))  # type: ignore[arg-type]
    if rank == 0 or rank == 1:  # pylint: disable=consider-using-in
        result = self
    else:
        result = op.Transpose(self, perm=[1, 0])  # type: ignore[arg-type]
    return result


def aten_t_copy(self: TensorType) -> TensorType:
    # t_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_take(self: TensorType, index: TensorType) -> TensorType:
    # take(Tensor self, Tensor index) -> Tensor

    raise NotImplementedError()


def aten_take_along_dim(
    self: TensorType, indices: TensorType, dim: Optional[int] = None
) -> TensorType:
    # take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor

    raise NotImplementedError()


def aten_tan(self: TensorType) -> TensorType:
    # tan(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_tanh(self: TensorType) -> TensorType:
    # tanh(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_tensordot(
    self: TensorType, other: TensorType, dims_self: Sequence[int], dims_other: Sequence[int]
) -> TensorType:
    # tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor

    raise NotImplementedError()


def aten_threshold(self: TensorType, threshold: float, value: float) -> TensorType:
    # threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor

    raise NotImplementedError()


def aten_threshold_backward(
    grad_output: TensorType, self: TensorType, threshold: float
) -> TensorType:
    # threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor

    raise NotImplementedError()


def aten_tile(self: TensorType, dims: Sequence[int]) -> TensorType:
    # tile(Tensor self, int[] dims) -> Tensor

    raise NotImplementedError()


def aten_to_dense(self: TensorType, dtype: Optional[int] = None) -> TensorType:
    # to_dense(Tensor self, ScalarType? dtype=None) -> Tensor

    raise NotImplementedError()


def aten_to_dense_backward(grad: TensorType, input: TensorType) -> TensorType:
    # to_dense_backward(Tensor grad, Tensor input) -> Tensor

    raise NotImplementedError()


def aten_to_mkldnn(self: TensorType, dtype: Optional[int] = None) -> TensorType:
    # to_mkldnn(Tensor self, ScalarType? dtype=None) -> Tensor

    raise NotImplementedError()


def aten_to_mkldnn_backward(grad: TensorType, input: TensorType) -> TensorType:
    # to_mkldnn_backward(Tensor grad, Tensor input) -> Tensor

    raise NotImplementedError()


def aten_to_padded_tensor(
    self: TensorType, padding: float, output_size: Optional[INT64] = None
) -> TensorType:
    # to_padded_tensor(Tensor self, float padding, SymInt[]? output_size=None) -> Tensor

    raise NotImplementedError()


def aten_to_sparse(self: TensorType) -> TensorType:
    # to_sparse(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_to_sparse_bsc(self: TensorType, blocksize: Sequence[int]) -> TensorType:
    # to_sparse_bsc(Tensor self, int[2] blocksize) -> Tensor

    raise NotImplementedError()


def aten_to_sparse_bsr(self: TensorType, blocksize: Sequence[int]) -> TensorType:
    # to_sparse_bsr(Tensor self, int[2] blocksize) -> Tensor

    raise NotImplementedError()


def aten_to_sparse_csc(self: TensorType) -> TensorType:
    # to_sparse_csc(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_to_sparse_csr(self: TensorType) -> TensorType:
    # to_sparse_csr(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_topk(
    self: TensorType, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
) -> tuple[TensorType, TensorType]:
    # topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)

    raise NotImplementedError()


def aten_trace(self: TensorType) -> TensorType:
    # trace(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_trace_backward(grad: TensorType, sizes: INT64) -> TensorType:
    # trace_backward(Tensor grad, SymInt[] sizes) -> Tensor

    raise NotImplementedError()


def aten_transpose(self, dim0: int, dim1: int):
    # transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)

    # FIXME(justinchuby): onnxscript raises Unsupported expression type
    return op.Transpose(self, [dim0, dim1])


def aten_triangular_solve(
    self: TensorType,
    A: TensorType,
    upper: bool = True,
    transpose: bool = False,
    unitriangular: bool = False,
) -> tuple[TensorType, TensorType]:
    # triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)

    raise NotImplementedError()


def aten_tril(self: TensorType, diagonal: int = 0) -> TensorType:
    # tril(Tensor self, int diagonal=0) -> Tensor

    raise NotImplementedError()


def aten_tril_indices(row: int, col: int, offset: int = 0) -> TensorType:
    # tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_triplet_margin_loss(
    anchor: TensorType,
    positive: TensorType,
    negative: TensorType,
    margin: float = 1.0,
    p: float = 2,
    eps: float = 1e-06,
    swap: bool = False,
    reduction: int = 1,
) -> TensorType:
    # triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor

    raise NotImplementedError()


def aten_triu(self: TensorType, diagonal: int = 0) -> TensorType:
    # triu(Tensor self, int diagonal=0) -> Tensor

    raise NotImplementedError()


def aten_triu_indices(row: int, col: int, offset: int = 0) -> TensorType:
    # triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_true_divide(self: TensorType, other: TensorType) -> TensorType:
    # true_divide.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_trunc(self: TensorType) -> TensorType:
    # trunc(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_type_as(self: TensorType, other: TensorType) -> TensorType:
    # type_as(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_unfold(self: TensorType, dimension: int, size: int, step: int) -> TensorType:
    # unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)

    raise NotImplementedError()


def aten_unfold_backward(
    grad_in: TensorType, input_sizes: INT64, dim: int, size: int, step: int
) -> TensorType:
    # unfold_backward(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step) -> Tensor

    raise NotImplementedError()


def aten_unfold_copy(self: TensorType, dimension: int, size: int, step: int) -> TensorType:
    # unfold_copy(Tensor self, int dimension, int size, int step) -> Tensor

    raise NotImplementedError()


def aten_unique_consecutive(
    self: TensorType,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: Optional[int] = None,
) -> tuple[TensorType, TensorType, TensorType]:
    # unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_unique_dim(
    self: TensorType,
    dim: int,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple[TensorType, TensorType, TensorType]:
    # unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_unique_dim_consecutive(
    self: TensorType, dim: int, return_inverse: bool = False, return_counts: bool = False
) -> tuple[TensorType, TensorType, TensorType]:
    # unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)

    raise NotImplementedError()


def aten_unsafe_chunk(self: TensorType, chunks: int, dim: int = 0) -> TensorType:
    # unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]

    raise NotImplementedError()


def aten_unsafe_split(self: TensorType, split_size: INT64, dim: int = 0) -> TensorType:
    # unsafe_split.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]

    raise NotImplementedError()


def aten_unsafe_split_with_sizes(
    self: TensorType, split_sizes: INT64, dim: int = 0
) -> TensorType:
    # unsafe_split_with_sizes(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]

    raise NotImplementedError()


def aten_unsqueeze(self: TensorType, dim: int) -> TensorType:
    # unsqueeze(Tensor(a) self, int dim) -> Tensor(a)

    raise NotImplementedError()


def aten_unsqueeze_copy(self: TensorType, dim: int) -> TensorType:
    # unsqueeze_copy(Tensor self, int dim) -> Tensor

    raise NotImplementedError()


def aten_value_selecting_reduction_backward(
    grad: TensorType, dim: int, indices: TensorType, sizes: INT64, keepdim: bool
) -> TensorType:
    # value_selecting_reduction_backward(Tensor grad, int dim, Tensor indices, SymInt[] sizes, bool keepdim) -> Tensor

    raise NotImplementedError()


def aten_values(self: TensorType) -> TensorType:
    # values(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_values_copy(self: TensorType) -> TensorType:
    # values_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_vander(
    x: TensorType, N: Optional[int] = None, increasing: bool = False
) -> TensorType:
    # vander(Tensor x, int? N=None, bool increasing=False) -> Tensor

    raise NotImplementedError()


def aten_var(self: TensorType, unbiased: bool = True) -> TensorType:
    # var(Tensor self, bool unbiased=True) -> Tensor

    raise NotImplementedError()


def aten_var_mean(self: TensorType, unbiased: bool = True) -> tuple[TensorType, TensorType]:
    # var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)

    raise NotImplementedError()


def aten_vdot(self: TensorType, other: TensorType) -> TensorType:
    # vdot(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_view(self: TensorType, size: INT64) -> TensorType:
    # view(Tensor(a) self, SymInt[] size) -> Tensor(a)

    raise NotImplementedError()


def aten_view_as(self: TensorType, other: TensorType) -> TensorType:
    # view_as(Tensor(a) self, Tensor other) -> Tensor(a)

    raise NotImplementedError()


def aten_view_as_complex(self: TensorType) -> TensorType:
    # view_as_complex(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_view_as_complex_copy(self: TensorType) -> TensorType:
    # view_as_complex_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_view_as_real(self: TensorType) -> TensorType:
    # view_as_real(Tensor(a) self) -> Tensor(a)

    raise NotImplementedError()


def aten_view_as_real_copy(self: TensorType) -> TensorType:
    # view_as_real_copy(Tensor self) -> Tensor

    raise NotImplementedError()


def aten_view_copy(self: TensorType, size: INT64) -> TensorType:
    # view_copy(Tensor self, SymInt[] size) -> Tensor

    raise NotImplementedError()


def aten_vstack(tensors: Sequence[TensorType]) -> TensorType:
    # vstack(Tensor[] tensors) -> Tensor

    raise NotImplementedError()


def aten_where(condition: TensorType) -> TensorType:
    # where(Tensor condition) -> Tensor[]

    raise NotImplementedError()


def aten_xlogy(self: TensorType, other: TensorType) -> TensorType:
    # xlogy.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_xor(self: TensorType, other: TensorType) -> TensorType:
    # __xor__.Tensor(Tensor self, Tensor other) -> Tensor

    raise NotImplementedError()


def aten_zeros(size: INT64) -> TensorType:
    # zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    raise NotImplementedError()


def aten_zeros_like(self: TensorType, memory_format: Optional[str] = None) -> TensorType:
    # zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    raise NotImplementedError()
