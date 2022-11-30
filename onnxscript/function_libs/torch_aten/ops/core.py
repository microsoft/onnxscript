# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""torch.ops.aten operators under the `core` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from onnxscript import INT64, TensorType


def aten_abs(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_absolute(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_acos(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_acosh(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_adaptive_avg_pool1d(self: TensorType, output_size: Sequence[int]) -> TensorType:
    raise NotImplementedError()


def aten_adaptive_max_pool1d(
    self: TensorType, output_size: Sequence[int]
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_addbmm(
    self: TensorType, batch1: TensorType, batch2: TensorType, *, beta: int = 1, alpha: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_addcdiv(
    self: TensorType, tensor1: TensorType, tensor2: TensorType, *, value: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_addcmul(
    self: TensorType, tensor1: TensorType, tensor2: TensorType, *, value: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_addmm(
    self: TensorType, mat1: TensorType, mat2: TensorType, *, beta: int = 1, alpha: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_addmv(
    self: TensorType, mat: TensorType, vec: TensorType, *, beta: int = 1, alpha: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_addr(
    self: TensorType, vec1: TensorType, vec2: TensorType, *, beta: int = 1, alpha: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_adjoint(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_affine_grid_generator(
    theta: TensorType, size: Sequence[int], align_corners: bool
) -> TensorType:
    raise NotImplementedError()


def aten_affine_grid_generator_backward(
    grad: TensorType, size: Sequence[int], align_corners: bool
) -> TensorType:
    raise NotImplementedError()


def aten_alias(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_alias_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_align_as(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_align_tensors(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_align_to(self: TensorType, names: Sequence[str]) -> TensorType:
    raise NotImplementedError()


def aten_all(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_allclose(
    self: TensorType,
    other: TensorType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    raise NotImplementedError()


def aten_alpha_dropout(input: TensorType, p: float, train: bool) -> TensorType:
    raise NotImplementedError()


def aten_amax(self: TensorType, dim: Sequence[int] = (), keepdim: bool = False) -> TensorType:
    raise NotImplementedError()


def aten_amin(self: TensorType, dim: Sequence[int] = (), keepdim: bool = False) -> TensorType:
    raise NotImplementedError()


def aten_aminmax(
    self: TensorType, *, dim: Optional[int] = None, keepdim: bool = False
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_angle(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_any(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_arange(end: float) -> TensorType:
    raise NotImplementedError()


def aten_arccos(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_arccosh(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_arcsin(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_arcsinh(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_arctan(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_arctan2(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_arctanh(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_argmax(
    self: TensorType, dim: Optional[int] = None, keepdim: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_argmin(
    self: TensorType, dim: Optional[int] = None, keepdim: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_argsort(self: TensorType, dim: int = -1, descending: bool = False) -> TensorType:
    raise NotImplementedError()


def aten_argwhere(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_as_strided(
    self: TensorType,
    size: INT64[...],
    stride: INT64[...],
    storage_offset: Optional[INT64] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_as_strided_copy(
    self: TensorType,
    size: INT64[...],
    stride: INT64[...],
    storage_offset: Optional[INT64] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_as_strided_scatter(
    self: TensorType,
    src: TensorType,
    size: INT64[...],
    stride: INT64[...],
    storage_offset: Optional[INT64] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_asin(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_asinh(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_atan(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_atan2(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_atanh(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_atleast_1d(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_atleast_2d(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_atleast_3d(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_avg_pool1d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> TensorType:
    raise NotImplementedError()


def aten_baddbmm(
    self: TensorType, batch1: TensorType, batch2: TensorType, *, beta: int = 1, alpha: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_bartlett_window(window_length: int) -> TensorType:
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
    raise NotImplementedError()


def aten_batch_norm_elemt(
    input: TensorType,
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    mean: TensorType,
    invstd: TensorType,
    eps: float,
) -> TensorType:
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
    raise NotImplementedError()


def aten_batch_norm_stats(input: TensorType, eps: float) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_batch_norm_update_stats(
    input: TensorType,
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    momentum: float,
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_bernoulli(self: TensorType, *, generator: Optional[str] = None) -> TensorType:
    raise NotImplementedError()


def aten_bilinear(
    input1: TensorType,
    input2: TensorType,
    weight: TensorType,
    bias: Optional[TensorType] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_binary_cross_entropy_with_logits(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    pos_weight: Optional[TensorType] = None,
    reduction: int = "mean",
) -> TensorType:
    raise NotImplementedError()


def aten_bincount(
    self: TensorType, weights: Optional[TensorType] = None, minlength: int = 0
) -> TensorType:
    raise NotImplementedError()


def aten_binomial(
    count: TensorType, prob: TensorType, generator: Optional[str] = None
) -> TensorType:
    raise NotImplementedError()


def aten_bitwise_not(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_blackman_window(window_length: int) -> TensorType:
    raise NotImplementedError()


def aten_block_diag(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_bmm(self: TensorType, mat2: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_broadcast_tensors(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_broadcast_to(self: TensorType, size: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_can_cast(from_: int, to: int) -> bool:
    raise NotImplementedError()


def aten_cartesian_prod(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_cat(tensors: TensorType[...], dim: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_ccol_indices(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_ccol_indices_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_cdist(
    x1: TensorType, x2: TensorType, p: float = 2, compute_mode: Optional[int] = None
) -> TensorType:
    raise NotImplementedError()


def aten_ceil(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_celu(self: TensorType, alpha: float = 1.0) -> TensorType:
    raise NotImplementedError()


def aten_chain_matmul(matrices: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_chalf(self: TensorType, *, memory_format: Optional[str] = None) -> TensorType:
    raise NotImplementedError()


def aten_channel_shuffle(self: TensorType, groups: int) -> TensorType:
    raise NotImplementedError()


def aten_cholesky(self: TensorType, upper: bool = False) -> TensorType:
    raise NotImplementedError()


def aten_cholesky_inverse(self: TensorType, upper: bool = False) -> TensorType:
    raise NotImplementedError()


def aten_cholesky_solve(
    self: TensorType, input2: TensorType, upper: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_choose_qparams_optimized(
    input: TensorType, numel: int, n_bins: int, ratio: float, bit_width: int
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_chunk(self: TensorType, chunks: int, dim: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_clamp(
    self: TensorType, min: Optional[float] = None, max: Optional[float] = None
) -> TensorType:
    raise NotImplementedError()


def aten_clamp_max(self: TensorType, max: float) -> TensorType:
    raise NotImplementedError()


def aten_clamp_min(self: TensorType, min: float) -> TensorType:
    raise NotImplementedError()


def aten_clip(
    self: TensorType, min: Optional[float] = None, max: Optional[float] = None
) -> TensorType:
    raise NotImplementedError()


def aten_clone(self: TensorType, *, memory_format: Optional[str] = None) -> TensorType:
    raise NotImplementedError()


def aten_coalesce(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_col_indices(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_col_indices_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_column_stack(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_combinations(
    self: TensorType, r: int = 2, with_replacement: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_complex(real: TensorType, imag: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_concat(tensors: TensorType[...], dim: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_concatenate(tensors: TensorType[...], dim: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_conj(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_conj_physical(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_constant_pad_nd(self: TensorType, pad: INT64[...], value: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_contiguous(
    self: TensorType, *, memory_format: str = "contiguous_format"
) -> TensorType:
    raise NotImplementedError()


def aten_conv1d(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = 1,
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    groups: int = 1,
) -> TensorType:
    raise NotImplementedError()


def aten_conv2d(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = 1,
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    groups: int = 1,
) -> TensorType:
    raise NotImplementedError()


def aten_conv3d(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = 1,
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    groups: int = 1,
) -> TensorType:
    raise NotImplementedError()


def aten_conv_tbc(
    self: TensorType, weight: TensorType, bias: TensorType, pad: int = 0
) -> TensorType:
    raise NotImplementedError()


def aten_conv_tbc_backward(
    self: TensorType, input: TensorType, weight: TensorType, bias: TensorType, pad: int
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_conv_transpose1d(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType] = None,
    stride: Sequence[int] = 1,
    padding: Sequence[int] = 0,
    output_padding: Sequence[int] = 0,
    groups: int = 1,
    dilation: Sequence[int] = 1,
) -> TensorType:
    raise NotImplementedError()


def aten_convolution(
    input: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    stride: Sequence[int],
    padding: INT64[...],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: INT64[...],
    groups: int,
) -> TensorType:
    raise NotImplementedError()


def aten_convolution_backward(
    grad_output: TensorType,
    input: TensorType,
    weight: TensorType,
    bias_sizes: Optional[INT64],
    stride: Sequence[int],
    padding: INT64[...],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: INT64[...],
    groups: int,
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType, TensorType]:
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
    raise NotImplementedError()


def aten_copy(self: TensorType, src: TensorType, non_blocking: bool = False) -> TensorType:
    raise NotImplementedError()


def aten_corrcoef(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_cos(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_cosh(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_cosine_embedding_loss(
    input1: TensorType,
    input2: TensorType,
    target: TensorType,
    margin: float = 0.0,
    reduction: int = "mean",
) -> TensorType:
    raise NotImplementedError()


def aten_cosine_similarity(
    x1: TensorType, x2: TensorType, dim: int = 1, eps: float = 1e-08
) -> TensorType:
    raise NotImplementedError()


def aten_count_nonzero(self: TensorType, dim: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_cov(
    self: TensorType,
    *,
    correction: int = 1,
    fweights: Optional[TensorType] = None,
    aweights: Optional[TensorType] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_cross(self: TensorType, other: TensorType, dim: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_crow_indices(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_crow_indices_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_cudnn_affine_grid_generator(
    theta: TensorType, N: int, C: int, H: int, W: int
) -> TensorType:
    raise NotImplementedError()


def aten_cudnn_affine_grid_generator_backward(
    grad: TensorType, N: int, C: int, H: int, W: int
) -> TensorType:
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
    raise NotImplementedError()


def aten_cudnn_grid_sampler(self: TensorType, grid: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_cudnn_grid_sampler_backward(
    self: TensorType, grid: TensorType, grad_output: TensorType
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_cudnn_is_acceptable(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_cummax(self: TensorType, dim: int) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_cummaxmin_backward(
    grad: TensorType, input: TensorType, indices: TensorType, dim: int
) -> TensorType:
    raise NotImplementedError()


def aten_cummin(self: TensorType, dim: int) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_cumprod(self: TensorType, dim: int, *, dtype: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_cumprod_backward(
    grad: TensorType, input: TensorType, dim: int, output: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_cumsum(self: TensorType, dim: int, *, dtype: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_data(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_deg2rad(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_dense_dim(self: TensorType) -> int:
    raise NotImplementedError()


def aten_det(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_detach(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_detach_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_diag(self: TensorType, diagonal: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_diag_embed(
    self: TensorType, offset: int = 0, dim1: int = -2, dim2: int = -1
) -> TensorType:
    raise NotImplementedError()


def aten_diagflat(self: TensorType, offset: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_diagonal(
    self: TensorType, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_diagonal_backward(
    grad_output: TensorType, input_sizes: INT64[...], offset: int, dim1: int, dim2: int
) -> TensorType:
    raise NotImplementedError()


def aten_diagonal_copy(
    self: TensorType, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_diagonal_scatter(
    self: TensorType, src: TensorType, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_diff(
    self: TensorType,
    n: int = 1,
    dim: int = -1,
    prepend: Optional[TensorType] = None,
    append: Optional[TensorType] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_digamma(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_dist(self: TensorType, other: TensorType, p: int = 2) -> TensorType:
    raise NotImplementedError()


def aten_dot(self: TensorType, tensor: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_dropout(input: TensorType, p: float, train: bool) -> TensorType:
    raise NotImplementedError()


def aten_dstack(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_einsum(
    equation: str, tensors: TensorType[...], *, path: Optional[int] = None
) -> TensorType:
    raise NotImplementedError()


def aten_embedding(
    weight: TensorType,
    indices: TensorType,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_embedding_backward(
    grad: TensorType,
    indices: TensorType,
    num_weights: INT64,
    padding_idx: int,
    scale_grad_by_freq: bool,
    sparse: bool,
) -> TensorType:
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
    raise NotImplementedError()


def aten_embedding_dense_backward(
    grad_output: TensorType,
    indices: TensorType,
    num_weights: INT64,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> TensorType:
    raise NotImplementedError()


def aten_embedding_sparse_backward(
    grad: TensorType,
    indices: TensorType,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> TensorType:
    raise NotImplementedError()


def aten_empty_like(self: TensorType, *, memory_format: Optional[str] = None) -> TensorType:
    raise NotImplementedError()


def aten_empty_quantized(
    size: Sequence[int], qtensor: TensorType, *, memory_format: Optional[str] = None
) -> TensorType:
    raise NotImplementedError()


def aten_empty_strided(size: INT64[...], stride: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_equal(self: TensorType, other: TensorType) -> bool:
    raise NotImplementedError()


def aten_erf(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_erfc(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_erfinv(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_exp(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_exp2(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_expand(self: TensorType, size: INT64[...], *, implicit: bool = False) -> TensorType:
    raise NotImplementedError()


def aten_expand_as(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_expand_copy(
    self: TensorType, size: INT64[...], *, implicit: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_expm1(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_eye(n: int) -> TensorType:
    raise NotImplementedError()


def aten_fake_quantize_per_channel_affine(
    self: TensorType,
    scale: TensorType,
    zero_point: TensorType,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> TensorType:
    raise NotImplementedError()


def aten_fake_quantize_per_channel_affine_cachemask(
    self: TensorType,
    scale: TensorType,
    zero_point: TensorType,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_fake_quantize_per_channel_affine_cachemask_backward(
    grad: TensorType, mask: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_fake_quantize_per_tensor_affine(
    self: TensorType, scale: float, zero_point: int, quant_min: int, quant_max: int
) -> TensorType:
    raise NotImplementedError()


def aten_fake_quantize_per_tensor_affine_cachemask(
    self: TensorType, scale: float, zero_point: int, quant_min: int, quant_max: int
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_fake_quantize_per_tensor_affine_cachemask_backward(
    grad: TensorType, mask: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_fbgemm_linear_fp16_weight(
    input: TensorType, packed_weight: TensorType, bias: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_fbgemm_linear_fp16_weight_fp32_activation(
    input: TensorType, packed_weight: TensorType, bias: TensorType
) -> TensorType:
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
    raise NotImplementedError()


def aten_fbgemm_linear_quantize_weight(
    input: TensorType,
) -> tuple[TensorType, TensorType, float, int]:
    raise NotImplementedError()


def aten_fbgemm_pack_gemm_matrix_fp16(input: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_fbgemm_pack_quantized_matrix(input: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_feature_alpha_dropout(input: TensorType, p: float, train: bool) -> TensorType:
    raise NotImplementedError()


def aten_feature_dropout(input: TensorType, p: float, train: bool) -> TensorType:
    raise NotImplementedError()


def aten_fix(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_flip(self: TensorType, dims: Sequence[int]) -> TensorType:
    raise NotImplementedError()


def aten_fliplr(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_flipud(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_floor(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_floor_divide(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_fmax(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_fmin(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_frac(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_frobenius_norm(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_from_file(
    filename: str, shared: Optional[bool] = None, size: Optional[int] = 0
) -> TensorType:
    raise NotImplementedError()


def aten_full(size: INT64[...], fill_value: float) -> TensorType:
    raise NotImplementedError()


def aten_full_like(
    self: TensorType, fill_value: float, *, memory_format: Optional[str] = None
) -> TensorType:
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
    raise NotImplementedError()


def aten_gather(
    self: TensorType, dim: int, index: TensorType, *, sparse_grad: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_gather_backward(
    grad: TensorType, self: TensorType, dim: int, index: TensorType, sparse_grad: bool
) -> TensorType:
    raise NotImplementedError()


def aten_gcd(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_geqrf(self: TensorType) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_ger(self: TensorType, vec2: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_grid_sampler(
    input: TensorType,
    grid: TensorType,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TensorType:
    raise NotImplementedError()


def aten_grid_sampler_2d(
    input: TensorType,
    grid: TensorType,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TensorType:
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
    raise NotImplementedError()


def aten_grid_sampler_3d(
    input: TensorType,
    grid: TensorType,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TensorType:
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
    raise NotImplementedError()


def aten_group_norm(
    input: TensorType,
    num_groups: int,
    weight: Optional[TensorType] = None,
    bias: Optional[TensorType] = None,
    eps: float = 1e-05,
    cudnn_enabled: bool = True,
) -> TensorType:
    raise NotImplementedError()


def aten_gru_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_hamming_window(window_length: int) -> TensorType:
    raise NotImplementedError()


def aten_hann_window(window_length: int) -> TensorType:
    raise NotImplementedError()


def aten_hardshrink(self: TensorType, lambd: float = 0.5) -> TensorType:
    raise NotImplementedError()


def aten_hardshrink_backward(
    grad_out: TensorType, self: TensorType, lambd: float
) -> TensorType:
    raise NotImplementedError()


def aten_heaviside(self: TensorType, values: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_hinge_embedding_loss(
    self: TensorType, target: TensorType, margin: float = 1.0, reduction: int = "mean"
) -> TensorType:
    raise NotImplementedError()


def aten_histc(self: TensorType, bins: int = 100, min: int = 0, max: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_histogramdd(
    self: TensorType,
    bins: Sequence[int],
    range: Optional[float] = None,
    weight: Optional[TensorType] = None,
    density: bool = False,
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_hspmm(mat1: TensorType, mat2: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_hstack(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_hypot(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_i0(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_igamma(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_igammac(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_imag(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_index_add(
    self: TensorType, dim: int, index: TensorType, source: TensorType, *, alpha: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_index_copy(
    self: TensorType, dim: int, index: TensorType, source: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_index_put(
    self: TensorType,
    indices: Optional[TensorType[...]],
    values: TensorType,
    accumulate: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_index_reduce(
    self: TensorType,
    dim: int,
    index: TensorType,
    source: TensorType,
    reduce: str,
    *,
    include_self: bool = True,
) -> TensorType:
    raise NotImplementedError()


def aten_index_select(self: TensorType, dim: int, index: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_index_select_backward(
    grad: TensorType, self_sizes: INT64[...], dim: int, index: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_indices(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_indices_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_inner(self: TensorType, other: TensorType) -> TensorType:
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
    raise NotImplementedError()


def aten_int_repr(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_inverse(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_is_coalesced(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_complex(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_conj(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_distributed(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_floating_point(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_inference(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_leaf(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_neg(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_nonzero(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_pinned(self: TensorType, device: Optional[str] = None) -> bool:
    raise NotImplementedError()


def aten_is_same_size(self: TensorType, other: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_set_to(self: TensorType, tensor: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_signed(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_is_vulkan_available() -> bool:
    raise NotImplementedError()


def aten_isclose(
    self: TensorType,
    other: TensorType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_isfinite(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_isinf(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_isnan(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_isneginf(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_isposinf(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_isreal(self: TensorType) -> TensorType:
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
    raise NotImplementedError()


def aten_item(self: TensorType) -> float:
    raise NotImplementedError()


def aten_kaiser_window(window_length: int) -> TensorType:
    raise NotImplementedError()


def aten_kl_div(
    self: TensorType, target: TensorType, reduction: int = "mean", *, log_target: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_kron(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_kthvalue(
    self: TensorType, k: int, dim: int = -1, keepdim: bool = False
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_layer_norm(
    input: TensorType,
    normalized_shape: Sequence[int],
    weight: Optional[TensorType] = None,
    bias: Optional[TensorType] = None,
    eps: float = 1e-05,
    cudnn_enable: bool = True,
) -> TensorType:
    raise NotImplementedError()


def aten_lcm(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_lgamma(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_lift(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_lift_fresh(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_lift_fresh_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_linear_backward(
    self: TensorType, grad_output: TensorType, weight: TensorType, output_mask: Sequence[bool]
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_linspace(start: float, end: float, steps: int) -> TensorType:
    raise NotImplementedError()


def aten_log(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_log10(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_log1p(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_log2(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_logaddexp(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_logaddexp2(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_logcumsumexp(self: TensorType, dim: int) -> TensorType:
    raise NotImplementedError()


def aten_logdet(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_logical_and(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_logical_not(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_logical_or(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_logical_xor(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_logit(self: TensorType, eps: Optional[float] = None) -> TensorType:
    raise NotImplementedError()


def aten_logspace(start: float, end: float, steps: int, base: float = 10.0) -> TensorType:
    raise NotImplementedError()


def aten_logsumexp(self: TensorType, dim: Sequence[int], keepdim: bool = False) -> TensorType:
    raise NotImplementedError()


def aten_lstm_cell(
    input: TensorType,
    hx: TensorType[...],
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_lstm_mps_backward(
    grad_y: TensorType,
    grad_hy: Optional[TensorType],
    grad_cy: Optional[TensorType],
    z_state: TensorType,
    cell_state_fwd: TensorType,
    input: TensorType,
    hx: TensorType[...],
    params: TensorType[...],
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_lu_solve(self: TensorType, LU_data: TensorType, LU_pivots: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_lu_unpack(
    LU_data: TensorType,
    LU_pivots: TensorType,
    unpack_data: bool = True,
    unpack_pivots: bool = True,
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_mH(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_mT(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_margin_ranking_loss(
    input1: TensorType,
    input2: TensorType,
    target: TensorType,
    margin: float = 0.0,
    reduction: int = "mean",
) -> TensorType:
    raise NotImplementedError()


def aten_masked_scatter(self: TensorType, mask: TensorType, source: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_masked_select(self: TensorType, mask: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_masked_select_backward(
    grad: TensorType, input: TensorType, mask: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_matmul(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_matmul_backward(
    grad: TensorType, self: TensorType, other: TensorType, mask: Sequence[bool]
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_matrix_H(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_matrix_exp(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_matrix_exp_backward(self: TensorType, grad: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_matrix_power(self: TensorType, n: int) -> TensorType:
    raise NotImplementedError()


def aten_max(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_max_pool1d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_max_pool1d_with_indices(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_max_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_max_pool3d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_maximum(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_mean(self: TensorType, *, dtype: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_median(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_meshgrid(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_min(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_minimum(self: TensorType, other: TensorType) -> TensorType:
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
    raise NotImplementedError()


def aten_miopen_convolution(
    self: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    padding: INT64[...],
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    benchmark: bool,
    deterministic: bool,
) -> TensorType:
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
    raise NotImplementedError()


def aten_miopen_convolution_transpose(
    self: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    padding: INT64[...],
    output_padding: INT64[...],
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    benchmark: bool,
    deterministic: bool,
) -> TensorType:
    raise NotImplementedError()


def aten_miopen_depthwise_convolution(
    self: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    padding: INT64[...],
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    benchmark: bool,
    deterministic: bool,
) -> TensorType:
    raise NotImplementedError()


def aten_miopen_rnn(
    input: TensorType,
    weight: TensorType[...],
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
    raise NotImplementedError()


def aten_miopen_rnn_backward(
    input: TensorType,
    weight: TensorType[...],
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
    raise NotImplementedError()


def aten_mkldnn_adaptive_avg_pool2d(
    self: TensorType, output_size: Sequence[int]
) -> TensorType:
    raise NotImplementedError()


def aten_mkldnn_adaptive_avg_pool2d_backward(
    grad_output: TensorType, self: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_mkldnn_convolution(
    self: TensorType,
    weight: TensorType,
    bias: Optional[TensorType],
    padding: INT64[...],
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> TensorType:
    raise NotImplementedError()


def aten_mkldnn_linear_backward(
    self: TensorType, grad_output: TensorType, weight: TensorType, output_mask: Sequence[bool]
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_mkldnn_linear_backward_input(
    input_size: Sequence[int], grad_output: TensorType, weight: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_mkldnn_linear_backward_weights(
    grad_output: TensorType, input: TensorType, weight: TensorType, bias_defined: bool
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_mkldnn_max_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_mkldnn_max_pool2d_backward(
    grad_output: TensorType,
    output: TensorType,
    input: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_mkldnn_max_pool3d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_mkldnn_max_pool3d_backward(
    grad_output: TensorType,
    output: TensorType,
    input: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_mm(self: TensorType, mat2: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_mode(
    self: TensorType, dim: int = -1, keepdim: bool = False
) -> tuple[TensorType, TensorType]:
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
    raise NotImplementedError()


def aten_mps_max_pool2d_backward(
    grad_output: TensorType,
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_msort(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_multinomial(
    self: TensorType,
    num_samples: int,
    replacement: bool = False,
    *,
    generator: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_mv(self: TensorType, vec: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_mvlgamma(self: TensorType, p: int) -> TensorType:
    raise NotImplementedError()


def aten_nan_to_num(
    self: TensorType,
    nan: Optional[float] = None,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_nanmean(
    self: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_nanmedian(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_nanquantile(
    self: TensorType,
    q: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    interpolation: str = "linear",
) -> TensorType:
    raise NotImplementedError()


def aten_nansum(
    self: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_narrow(self: TensorType, dim: int, start: INT64, length: INT64) -> TensorType:
    raise NotImplementedError()


def aten_narrow_copy(self: TensorType, dim: int, start: INT64, length: INT64) -> TensorType:
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
    raise NotImplementedError()


def aten_native_channel_shuffle(self: TensorType, groups: int) -> TensorType:
    raise NotImplementedError()


def aten_native_dropout(
    input: TensorType, p: float, train: Optional[bool]
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_native_dropout_backward(
    grad_output: TensorType, mask: TensorType, scale: float
) -> TensorType:
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
    raise NotImplementedError()


def aten_native_layer_norm(
    input: TensorType,
    normalized_shape: INT64[...],
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    eps: float,
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_native_layer_norm_backward(
    grad_out: TensorType,
    input: TensorType,
    normalized_shape: INT64[...],
    mean: TensorType,
    rstd: TensorType,
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_native_norm(self: TensorType, p: int = 2) -> TensorType:
    raise NotImplementedError()


def aten_neg(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_negative(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_new_empty(self: TensorType, size: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_new_empty_strided(
    self: TensorType, size: INT64[...], stride: INT64[...]
) -> TensorType:
    raise NotImplementedError()


def aten_new_full(self: TensorType, size: INT64[...], fill_value: float) -> TensorType:
    raise NotImplementedError()


def aten_new_ones(self: TensorType, size: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_new_zeros(self: TensorType, size: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_nextafter(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_nonzero(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_nonzero_numpy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_norm_except_dim(v: TensorType, pow: int = 2, dim: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_normal(
    self: TensorType, mean: float = 0, std: float = 1, *, generator: Optional[str] = None
) -> TensorType:
    raise NotImplementedError()


def aten_nuclear_norm(self: TensorType, keepdim: bool = False) -> TensorType:
    raise NotImplementedError()


def aten_numpy_T(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_ones(size: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_ones_like(self: TensorType, *, memory_format: Optional[str] = None) -> TensorType:
    raise NotImplementedError()


def aten_orgqr(self: TensorType, input2: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_ormqr(
    self: TensorType,
    input2: TensorType,
    input3: TensorType,
    left: bool = True,
    transpose: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_outer(self: TensorType, vec2: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_output_nr(self: TensorType) -> int:
    raise NotImplementedError()


def aten_pairwise_distance(
    x1: TensorType, x2: TensorType, p: float = 2, eps: float = 1e-06, keepdim: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_pdist(self: TensorType, p: float = 2) -> TensorType:
    raise NotImplementedError()


def aten_permute(self: TensorType, dims: Sequence[int]) -> TensorType:
    raise NotImplementedError()


def aten_permute_copy(self: TensorType, dims: Sequence[int]) -> TensorType:
    raise NotImplementedError()


def aten_pin_memory(self: TensorType, device: Optional[str] = None) -> TensorType:
    raise NotImplementedError()


def aten_pinverse(self: TensorType, rcond: float = 1e-15) -> TensorType:
    raise NotImplementedError()


def aten_pixel_shuffle(self: TensorType, upscale_factor: int) -> TensorType:
    raise NotImplementedError()


def aten_pixel_unshuffle(self: TensorType, downscale_factor: int) -> TensorType:
    raise NotImplementedError()


def aten_poisson(self: TensorType, generator: Optional[str] = None) -> TensorType:
    raise NotImplementedError()


def aten_poisson_nll_loss(
    input: TensorType,
    target: TensorType,
    log_input: bool,
    full: bool,
    eps: float,
    reduction: int,
) -> TensorType:
    raise NotImplementedError()


def aten_polar(abs: TensorType, angle: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_polygamma(n: int, self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_positive(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_prelu(self: TensorType, weight: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_prelu_backward(
    grad_output: TensorType, self: TensorType, weight: TensorType
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_prod(self: TensorType, *, dtype: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_promote_types(type1: int, type2: int) -> int:
    raise NotImplementedError()


def aten_put(
    self: TensorType, index: TensorType, source: TensorType, accumulate: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_q_per_channel_axis(self: TensorType) -> int:
    raise NotImplementedError()


def aten_q_per_channel_scales(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_q_per_channel_zero_points(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_q_scale(self: TensorType) -> float:
    raise NotImplementedError()


def aten_q_zero_point(self: TensorType) -> int:
    raise NotImplementedError()


def aten_qr(self: TensorType, some: bool = True) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_qscheme(self: TensorType) -> str:
    raise NotImplementedError()


def aten_quantile(
    self: TensorType,
    q: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    interpolation: str = "linear",
) -> TensorType:
    raise NotImplementedError()


def aten_quantize_per_channel(
    self: TensorType, scales: TensorType, zero_points: TensorType, axis: int, dtype: int
) -> TensorType:
    raise NotImplementedError()


def aten_quantize_per_tensor(
    self: TensorType, scale: float, zero_point: int, dtype: int
) -> TensorType:
    raise NotImplementedError()


def aten_quantize_per_tensor_dynamic(
    self: TensorType, dtype: int, reduce_range: bool
) -> TensorType:
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
    raise NotImplementedError()


def aten_quantized_lstm_cell(
    input: TensorType,
    hx: TensorType[...],
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
    raise NotImplementedError()


def aten_quantized_max_pool1d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_quantized_max_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = 0,
    dilation: Sequence[int] = 1,
    ceil_mode: bool = False,
) -> TensorType:
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
    raise NotImplementedError()


def aten_rad2deg(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_rand(size: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_rand_like(self: TensorType, *, memory_format: Optional[str] = None) -> TensorType:
    raise NotImplementedError()


def aten_randint(high: int, size: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_randint_like(
    self: TensorType, high: int, *, memory_format: Optional[str] = None
) -> TensorType:
    raise NotImplementedError()


def aten_randn(size: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_randn_like(self: TensorType, *, memory_format: Optional[str] = None) -> TensorType:
    raise NotImplementedError()


def aten_randperm(n: int) -> TensorType:
    raise NotImplementedError()


def aten_range(start: float, end: float) -> TensorType:
    raise NotImplementedError()


def aten_ravel(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_real(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_reciprocal(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_record_stream(self: TensorType, s: str) -> Any:
    raise NotImplementedError()


def aten_refine_names(self: TensorType, names: Sequence[str]) -> TensorType:
    raise NotImplementedError()


def aten_relu(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_rename(self: TensorType, names: Optional[str]) -> TensorType:
    raise NotImplementedError()


def aten_renorm(self: TensorType, p: float, dim: int, maxnorm: float) -> TensorType:
    raise NotImplementedError()


def aten_repeat(self: TensorType, repeats: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_reshape(self: TensorType, shape: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_reshape_as(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_resolve_conj(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_resolve_neg(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_retain_grad(self: TensorType) -> Any:
    raise NotImplementedError()


def aten_retains_grad(self: TensorType) -> bool:
    raise NotImplementedError()


def aten_rnn_relu_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_rnn_tanh_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_roll(self: TensorType, shifts: Sequence[int], dims: Sequence[int] = ()) -> TensorType:
    raise NotImplementedError()


def aten_rot90(self: TensorType, k: int = 1, dims: Sequence[int] = (0, 1)) -> TensorType:
    raise NotImplementedError()


def aten_round(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_row_indices(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_row_indices_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_row_stack(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_rrelu(
    self: TensorType,
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
    training: bool = False,
    generator: Optional[str] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_rsqrt(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_scalar_tensor(s: float) -> TensorType:
    raise NotImplementedError()


def aten_scatter_add(
    self: TensorType, dim: int, index: TensorType, src: TensorType
) -> TensorType:
    raise NotImplementedError()


def aten_segment_reduce(
    data: TensorType,
    reduce: str,
    *,
    lengths: Optional[TensorType] = None,
    indices: Optional[TensorType] = None,
    offsets: Optional[TensorType] = None,
    axis: int = 0,
    unsafe: bool = False,
    initial: Optional[float] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_select_backward(
    grad_output: TensorType, input_sizes: INT64[...], dim: int, index: int
) -> TensorType:
    raise NotImplementedError()


def aten_select_scatter(self: TensorType, src: TensorType, dim: int, index: int) -> TensorType:
    raise NotImplementedError()


def aten_selu(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_set_data(self: TensorType, new_data: TensorType) -> Any:
    raise NotImplementedError()


def aten_sgn(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_sigmoid(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_sign(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_signbit(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_sin(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_sinc(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_sinh(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_slice_backward(
    grad_output: TensorType,
    input_sizes: INT64[...],
    dim: int,
    start: INT64,
    end: INT64,
    step: INT64,
) -> TensorType:
    raise NotImplementedError()


def aten_slice_scatter(
    self: TensorType,
    src: TensorType,
    dim: int = 0,
    start: Optional[INT64] = None,
    end: Optional[INT64] = None,
    step: INT64 = 1,
) -> TensorType:
    raise NotImplementedError()


def aten_slogdet(self: TensorType) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_smm(self: TensorType, mat2: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_sort(
    self: TensorType, dim: int = -1, descending: bool = False
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_sparse_dim(self: TensorType) -> int:
    raise NotImplementedError()


def aten_sparse_mask(self: TensorType, mask: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_split_with_sizes(
    self: TensorType, split_sizes: INT64[...], dim: int = 0
) -> TensorType:
    raise NotImplementedError()


def aten_split_with_sizes_copy(
    self: TensorType, split_sizes: INT64[...], dim: int = 0
) -> TensorType:
    raise NotImplementedError()


def aten_sqrt(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_square(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_squeeze(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_squeeze_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_sspaddmm(
    self: TensorType, mat1: TensorType, mat2: TensorType, *, beta: int = 1, alpha: int = 1
) -> TensorType:
    raise NotImplementedError()


def aten_stack(tensors: TensorType[...], dim: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_std(self: TensorType, unbiased: bool = True) -> TensorType:
    raise NotImplementedError()


def aten_std_mean(self: TensorType, unbiased: bool = True) -> tuple[TensorType, TensorType]:
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
    raise NotImplementedError()


def aten_sum(self: TensorType, *, dtype: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_sum_to_size(self: TensorType, size: Sequence[int]) -> TensorType:
    raise NotImplementedError()


def aten_svd(
    self: TensorType, some: bool = True, compute_uv: bool = True
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_swapaxes(self: TensorType, axis0: int, axis1: int) -> TensorType:
    raise NotImplementedError()


def aten_swapdims(self: TensorType, dim0: int, dim1: int) -> TensorType:
    raise NotImplementedError()


def aten_symeig(
    self: TensorType, eigenvectors: bool = False, upper: bool = True
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_t(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_t_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_take(self: TensorType, index: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_take_along_dim(
    self: TensorType, indices: TensorType, dim: Optional[int] = None
) -> TensorType:
    raise NotImplementedError()


def aten_tan(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_tanh(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_tensordot(
    self: TensorType, other: TensorType, dims_self: Sequence[int], dims_other: Sequence[int]
) -> TensorType:
    raise NotImplementedError()


def aten_threshold(self: TensorType, threshold: float, value: float) -> TensorType:
    raise NotImplementedError()


def aten_threshold_backward(
    grad_output: TensorType, self: TensorType, threshold: float
) -> TensorType:
    raise NotImplementedError()


def aten_tile(self: TensorType, dims: Sequence[int]) -> TensorType:
    raise NotImplementedError()


def aten_to_dense(self: TensorType, dtype: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_to_dense_backward(grad: TensorType, input: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_to_mkldnn(self: TensorType, dtype: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_to_mkldnn_backward(grad: TensorType, input: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_to_padded_tensor(
    self: TensorType, padding: float, output_size: Optional[INT64] = None
) -> TensorType:
    raise NotImplementedError()


def aten_to_sparse(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_to_sparse_bsc(self: TensorType, blocksize: Sequence[int]) -> TensorType:
    raise NotImplementedError()


def aten_to_sparse_bsr(self: TensorType, blocksize: Sequence[int]) -> TensorType:
    raise NotImplementedError()


def aten_to_sparse_csc(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_to_sparse_csr(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_topk(
    self: TensorType, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_trace(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_trace_backward(grad: TensorType, sizes: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_triangular_solve(
    self: TensorType,
    A: TensorType,
    upper: bool = True,
    transpose: bool = False,
    unitriangular: bool = False,
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_tril(self: TensorType, diagonal: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_tril_indices(row: int, col: int, offset: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_triplet_margin_loss(
    anchor: TensorType,
    positive: TensorType,
    negative: TensorType,
    margin: float = 1.0,
    p: float = 2,
    eps: float = 1e-06,
    swap: bool = False,
    reduction: int = "mean",
) -> TensorType:
    raise NotImplementedError()


def aten_triu(self: TensorType, diagonal: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_triu_indices(row: int, col: int, offset: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_trunc(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_type_as(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_unfold(self: TensorType, dimension: int, size: int, step: int) -> TensorType:
    raise NotImplementedError()


def aten_unfold_backward(
    grad_in: TensorType, input_sizes: INT64[...], dim: int, size: int, step: int
) -> TensorType:
    raise NotImplementedError()


def aten_unfold_copy(self: TensorType, dimension: int, size: int, step: int) -> TensorType:
    raise NotImplementedError()


def aten_unique_consecutive(
    self: TensorType,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: Optional[int] = None,
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_unique_dim(
    self: TensorType,
    dim: int,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_unique_dim_consecutive(
    self: TensorType, dim: int, return_inverse: bool = False, return_counts: bool = False
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_unsafe_chunk(self: TensorType, chunks: int, dim: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_unsafe_split_with_sizes(
    self: TensorType, split_sizes: INT64[...], dim: int = 0
) -> TensorType:
    raise NotImplementedError()


def aten_unsqueeze(self: TensorType, dim: int) -> TensorType:
    raise NotImplementedError()


def aten_unsqueeze_copy(self: TensorType, dim: int) -> TensorType:
    raise NotImplementedError()


def aten_value_selecting_reduction_backward(
    grad: TensorType, dim: int, indices: TensorType, sizes: INT64[...], keepdim: bool
) -> TensorType:
    raise NotImplementedError()


def aten_values(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_values_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_vander(
    x: TensorType, N: Optional[int] = None, increasing: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_var(self: TensorType, unbiased: bool = True) -> TensorType:
    raise NotImplementedError()


def aten_var_mean(self: TensorType, unbiased: bool = True) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_vdot(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_view(self: TensorType, size: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_view_as(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_view_as_complex(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_view_as_complex_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_view_as_real(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_view_as_real_copy(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_view_copy(self: TensorType, size: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_vstack(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_where(condition: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_zeros(size: INT64[...]) -> TensorType:
    raise NotImplementedError()


def aten_zeros_like(self: TensorType, *, memory_format: Optional[str] = None) -> TensorType:
    raise NotImplementedError()
