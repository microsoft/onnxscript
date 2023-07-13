# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
"""torch.ops.aten operators under the `prims` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""
from __future__ import annotations

from typing import Optional, Sequence

from onnxscript import INT64
from onnxscript.function_libs.torch_lib.registration import torch_op
from onnxscript.function_libs.torch_lib.tensor_typing import TTensor
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType


def prims_abs(self: TensorType) -> TensorType:
    """abs(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_acos(self: TensorType) -> TensorType:
    """acos(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_acosh(self: TensorType) -> TensorType:
    """acosh(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_add(self: TensorType, other: TensorType) -> TensorType:
    """add(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_amax(
    inp: TensorType, dims: Optional[Sequence[int]], output_dtype: Optional[int] = None
) -> TensorType:
    """amax(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""

    raise NotImplementedError()


def prims_amin(
    inp: TensorType, dims: Optional[Sequence[int]], output_dtype: Optional[int] = None
) -> TensorType:
    """amin(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""

    raise NotImplementedError()


def prims_as_strided(
    a: TensorType, size: INT64, stride: INT64, storage_offset: INT64
) -> TensorType:
    """as_strided(Tensor a, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor"""

    raise NotImplementedError()


def prims_as_strided_scatter(
    self: TensorType, src: TensorType, size: INT64, stride: INT64, storage_offset: INT64
) -> TensorType:
    """as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor"""

    raise NotImplementedError()


def prims_asin(self: TensorType) -> TensorType:
    """asin(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_asinh(self: TensorType) -> TensorType:
    """asinh(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_atan(self: TensorType) -> TensorType:
    """atan(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_atan2(self: TensorType, other: TensorType) -> TensorType:
    """atan2(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_atanh(self: TensorType) -> TensorType:
    """atanh(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_bessel_i0(self: TensorType) -> TensorType:
    """bessel_i0(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_bessel_i0e(self: TensorType) -> TensorType:
    """bessel_i0e(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_bessel_i1(self: TensorType) -> TensorType:
    """bessel_i1(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_bessel_i1e(self: TensorType) -> TensorType:
    """bessel_i1e(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_bessel_j0(self: TensorType) -> TensorType:
    """bessel_j0(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_bessel_j1(self: TensorType) -> TensorType:
    """bessel_j1(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_bitwise_and(self: TensorType, other: TensorType) -> TensorType:
    """bitwise_and(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_bitwise_not(self: TensorType) -> TensorType:
    """bitwise_not(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_bitwise_or(self: TensorType, other: TensorType) -> TensorType:
    """bitwise_or(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_bitwise_xor(self: TensorType, other: TensorType) -> TensorType:
    """bitwise_xor(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_broadcast_in_dim(
    a: TensorType, shape: INT64, broadcast_dimensions: Sequence[int]
) -> TensorType:
    """broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)"""

    raise NotImplementedError()


def prims_cat(tensors: Sequence[TensorType], dim: int) -> TensorType:
    """cat(Tensor[] tensors, int dim) -> Tensor"""

    raise NotImplementedError()


def prims_cbrt(self: TensorType) -> TensorType:
    """cbrt(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_ceil(self: TensorType) -> TensorType:
    """ceil(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_clone(self: TensorType, memory_format: Optional[str] = None) -> TensorType:
    """clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""

    raise NotImplementedError()


def prims_collapse_view(a: TensorType, start: int, end: int) -> TensorType:
    """collapse_view(Tensor(a) a, int start, int end) -> Tensor(a)"""

    raise NotImplementedError()


def prims_conj(a: TensorType) -> TensorType:
    """conj(Tensor(a) a) -> Tensor(a)"""

    raise NotImplementedError()


def prims_conj_physical(self: TensorType) -> TensorType:
    """conj_physical(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("prims::convert_element_type")
def prims_convert_element_type(a: TensorType, dtype: int) -> TensorType:
    """convert_element_type(Tensor a, ScalarType dtype) -> Tensor"""

    return op.Cast(a, to=dtype)


def prims_copy_strided(a: TensorType, stride: INT64) -> TensorType:
    """copy_strided(Tensor a, SymInt[] stride) -> Tensor"""

    raise NotImplementedError()


def prims_copy_to(a: TensorType, b: TensorType) -> TensorType:
    """copy_to(Tensor a, Tensor b) -> Tensor"""

    raise NotImplementedError()


def prims_cos(self: TensorType) -> TensorType:
    """cos(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_cosh(self: TensorType) -> TensorType:
    """cosh(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("prims::device_put")
def prims_device_put(
    a: TTensor,
    device: str = "unspecified",  # pylint: disable=unused-argument
) -> TTensor:
    """device_put(Tensor a, Device device) -> Tensor"""

    # ONNX does not have the notion of a "device". So we just return the input
    return op.Identity(a)


def prims_digamma(self: TensorType) -> TensorType:
    """digamma(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_div(self: TensorType, other: TensorType) -> TensorType:
    """div(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_empty(shape: INT64, dtype: int, device: str, requires_grad: bool) -> TensorType:
    """empty(SymInt[] shape, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError()


def prims_empty_strided(
    shape: INT64, strides: INT64, dtype: int, device: str, requires_grad: bool
) -> TensorType:
    """empty_strided(SymInt[] shape, SymInt[] strides, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError()


def prims_eq(self: TensorType, other: TensorType) -> TensorType:
    """eq(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_erf(self: TensorType) -> TensorType:
    """erf(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_erf_inv(self: TensorType) -> TensorType:
    """erf_inv(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_erfc(self: TensorType) -> TensorType:
    """erfc(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_erfcx(self: TensorType) -> TensorType:
    """erfcx(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_exp(self: TensorType) -> TensorType:
    """exp(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_exp2(self: TensorType) -> TensorType:
    """exp2(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_expm1(self: TensorType) -> TensorType:
    """expm1(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_fft_c2c(self: TensorType, dim: Sequence[int], forward: bool) -> TensorType:
    """fft_c2c(Tensor self, *, int[] dim, bool forward) -> Tensor"""

    raise NotImplementedError()


def prims_fft_c2r(self: TensorType, dim: Sequence[int], last_dim_size: INT64) -> TensorType:
    """fft_c2r(Tensor self, *, int[] dim, SymInt last_dim_size) -> Tensor"""

    raise NotImplementedError()


def prims_fft_r2c(self: TensorType, dim: Sequence[int], onesided: bool) -> TensorType:
    """fft_r2c(Tensor self, *, int[] dim, bool onesided) -> Tensor"""

    raise NotImplementedError()


def prims_fill(self: TensorType, value: float) -> TensorType:
    """fill(Tensor self, Scalar value) -> Tensor"""

    raise NotImplementedError()


def prims_floor(self: TensorType) -> TensorType:
    """floor(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_fmax(self: TensorType, other: TensorType) -> TensorType:
    """fmax(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_fmin(self: TensorType, other: TensorType) -> TensorType:
    """fmin(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_fmod(self: TensorType, other: TensorType) -> TensorType:
    """fmod(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_full(
    shape: INT64, fill_value: float, dtype: int, device: str, requires_grad: bool
) -> TensorType:
    """full(SymInt[] shape, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError()


def prims_full_like(
    a: TensorType, fill_value: float, dtype: int, device: str, requires_grad: bool
) -> TensorType:
    """full_like(Tensor a, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError()


def prims_gcd(self: TensorType, other: TensorType) -> TensorType:
    """gcd(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_ge(self: TensorType, other: TensorType) -> TensorType:
    """ge(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_gt(self: TensorType, other: TensorType) -> TensorType:
    """gt(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_hypot(self: TensorType, other: TensorType) -> TensorType:
    """hypot(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_igamma(self: TensorType, other: TensorType) -> TensorType:
    """igamma(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_igammac(self: TensorType, other: TensorType) -> TensorType:
    """igammac(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_imag(self: TensorType) -> TensorType:
    """imag(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_iota(
    length: INT64, start: INT64, step: INT64, dtype: int, device: str, requires_grad: bool
) -> TensorType:
    """iota(SymInt length, *, SymInt start, SymInt step, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError()


def prims_isfinite(self: TensorType) -> TensorType:
    """isfinite(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_item(a: TensorType) -> float:
    """item(Tensor a) -> Scalar"""

    raise NotImplementedError()


def prims_le(self: TensorType, other: TensorType) -> TensorType:
    """le(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_lgamma(self: TensorType) -> TensorType:
    """lgamma(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_log(self: TensorType) -> TensorType:
    """log(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_log10(self: TensorType) -> TensorType:
    """log10(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_log1p(self: TensorType) -> TensorType:
    """log1p(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_log2(self: TensorType) -> TensorType:
    """log2(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_lt(self: TensorType, other: TensorType) -> TensorType:
    """lt(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_maximum(self: TensorType, other: TensorType) -> TensorType:
    """maximum(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_maximum_value(dtype: int) -> float:
    """maximum_value(ScalarType dtype) -> Scalar"""

    raise NotImplementedError()


def prims_minimum(self: TensorType, other: TensorType) -> TensorType:
    """minimum(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_minium_value(dtype: int) -> float:
    """minium_value(ScalarType dtype) -> Scalar"""

    raise NotImplementedError()


def prims_mul(self: TensorType, other: TensorType) -> TensorType:
    """mul(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_ndtri(self: TensorType) -> TensorType:
    """ndtri(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_ne(self: TensorType, other: TensorType) -> TensorType:
    """ne(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_neg(self: TensorType) -> TensorType:
    """neg(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_nextafter(self: TensorType, other: TensorType) -> TensorType:
    """nextafter(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_normal(
    shape: INT64, mean: float, std: float, dtype: int, device: str, requires_grad: bool
) -> TensorType:
    """normal(SymInt[] shape, *, Scalar mean, Scalar std, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError()


def prims_pow(self: TensorType, other: TensorType) -> TensorType:
    """pow(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_prod(
    inp: TensorType, dims: Optional[Sequence[int]], output_dtype: Optional[int] = None
) -> TensorType:
    """prod(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""

    raise NotImplementedError()


def prims_real(self: TensorType) -> TensorType:
    """real(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_reciprocal(self: TensorType) -> TensorType:
    """reciprocal(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_remainder(self: TensorType, other: TensorType) -> TensorType:
    """remainder(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_reshape(a: TensorType, shape: INT64) -> TensorType:
    """reshape(Tensor a, SymInt[] shape) -> Tensor"""

    raise NotImplementedError()


def prims_resize(a: TensorType, shape: INT64) -> TensorType:
    """resize(Tensor a, SymInt[] shape) -> Tensor"""

    raise NotImplementedError()


def prims_rev(a: TensorType, dims: Sequence[int]) -> TensorType:
    """rev(Tensor a, int[] dims) -> Tensor"""

    raise NotImplementedError()


def prims_round(self: TensorType) -> TensorType:
    """round(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_rsqrt(self: TensorType) -> TensorType:
    """rsqrt(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_scalar_tensor(
    s: float, dtype: Optional[int] = None, device: Optional[str] = None
) -> TensorType:
    """scalar_tensor(Scalar s, *, ScalarType? dtype=None, Device? device=None) -> Tensor"""

    raise NotImplementedError()


def prims_shift_left(self: TensorType, other: TensorType) -> TensorType:
    """shift_left(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_shift_right_arithmetic(self: TensorType, other: TensorType) -> TensorType:
    """shift_right_arithmetic(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_sign(self: TensorType) -> TensorType:
    """sign(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_signbit(self: TensorType) -> TensorType:
    """signbit(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_sin(self: TensorType) -> TensorType:
    """sin(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_sinh(self: TensorType) -> TensorType:
    """sinh(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_slice(
    a: TensorType, start_indices: INT64, limit_indices: INT64, strides: Optional[INT64] = None
) -> TensorType:
    """slice(Tensor(a) a, SymInt[] start_indices, SymInt[] limit_indices, SymInt[]? strides=None) -> Tensor(a)"""

    raise NotImplementedError()


def prims_slice_in_dim(
    a: TensorType, start_index: INT64, limit_index: INT64, stride: int = 1, axis: int = 0
) -> TensorType:
    """slice_in_dim(Tensor(a) a, SymInt start_index, SymInt limit_index, int stride=1, int axis=0) -> Tensor(a)"""

    raise NotImplementedError()


def prims_spherical_bessel_j0(self: TensorType) -> TensorType:
    """spherical_bessel_j0(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_split_dim(a: TensorType, dim: int, outer_length: INT64) -> TensorType:
    """split_dim(Tensor(a) a, int dim, SymInt outer_length) -> Tensor(a)"""

    raise NotImplementedError()


def prims_sqrt(self: TensorType) -> TensorType:
    """sqrt(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_squeeze(a: TensorType, dimensions: Sequence[int]) -> TensorType:
    """squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)"""

    raise NotImplementedError()


def prims_sub(self: TensorType, other: TensorType) -> TensorType:
    """sub(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def prims_sum(
    inp: TensorType, dims: Optional[Sequence[int]], output_dtype: Optional[int] = None
) -> TensorType:
    """sum(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""

    raise NotImplementedError()


def prims_svd(A: TensorType, full_matrices: bool) -> tuple[TensorType, TensorType, TensorType]:
    """svd(Tensor A, *, bool full_matrices) -> (Tensor U, Tensor S, Tensor Vh)"""

    raise NotImplementedError()


def prims_tan(self: TensorType) -> TensorType:
    """tan(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_tanh(self: TensorType) -> TensorType:
    """tanh(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_transpose(a: TensorType, permutation: Sequence[int]) -> TensorType:
    """transpose(Tensor(a) a, int[] permutation) -> Tensor(a)"""

    raise NotImplementedError()


def prims_trunc(self: TensorType) -> TensorType:
    """trunc(Tensor self) -> Tensor"""

    raise NotImplementedError()


def prims_uniform(
    shape: INT64, low: float, high: float, dtype: int, device: str
) -> TensorType:
    """uniform(SymInt[] shape, *, Scalar low, Scalar high, ScalarType dtype, Device device) -> Tensor"""

    raise NotImplementedError()


def prims_var(
    inp: TensorType,
    dims: Optional[Sequence[int]],
    correction: int,
    output_dtype: Optional[int] = None,
) -> TensorType:
    """var(Tensor inp, int[]? dims, *, int correction, ScalarType? output_dtype=None) -> Tensor"""

    raise NotImplementedError()


def prims_view_of(a: TensorType) -> TensorType:
    """view_of(Tensor(a) a) -> Tensor"""

    raise NotImplementedError()


def prims_where(pred: TensorType, a: TensorType, b: TensorType) -> TensorType:
    """where(Tensor pred, Tensor a, Tensor b) -> Tensor"""

    raise NotImplementedError()


def prims_zeta(self: TensorType, other: TensorType) -> TensorType:
    """zeta(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()
