# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
"""torch.ops.aten operators under the `core` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple, Union

from onnxscript import (
    BFLOAT16,
    BOOL,
    DOUBLE,
    FLOAT,
    FLOAT16,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    graph,
)
from onnxscript.function_libs.torch_lib.ops import common as common_ops
from onnxscript.function_libs.torch_lib.registration import torch_op
from onnxscript.function_libs.torch_lib.tensor_typing import (
    IntType,
    RealType,
    TFloat,
    TFloatHighPrecision,
    TFloatOrBFloat16,
    TInt,
    TReal,
    TRealOrUInt8,
    TRealUnlessFloat16OrInt8,
    TRealUnlessInt16OrInt8,
    TTensor,
    TTensor2,
    TTensorOrString,
)
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType

_INT64_MAX = 9223372036854775807
_INT64_MIN = -9223372036854775808
_MATH_PI = math.pi
IsScalar = common_ops.IsScalar
Rank = common_ops.Rank


@torch_op("aten::_local_scalar_dense")
def aten__local_scalar_dense(self: Union[FLOAT16, FLOAT, DOUBLE, BFLOAT16]) -> FLOAT:
    """_local_scalar_dense(Tensor self) -> Scalar"""

    # Return the first element in tensor as a scalar.
    return op.Cast(op.Gather(op.Reshape(self, [-1]), 0), to=FLOAT.dtype)


@torch_op("aten::_local_scalar_dense")
def aten__local_scalar_dense_int(self: IntType) -> INT64:
    """_local_scalar_dense(Tensor self) -> Scalar"""

    # Return the first element in tensor as a scalar.
    return op.Cast(op.Gather(op.Reshape(self, [-1]), 0), to=INT64.dtype)


@torch_op("aten::_log_softmax", trace_only=True)
def aten__log_softmax_half(
    self: Union[FLOAT16, BFLOAT16], dim: int, half_to_float: bool
) -> FLOAT:
    """_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""

    # trace_only because we need to cast conditionally based on half_to_float
    if half_to_float:
        self = op.Cast(self, to=FLOAT.dtype)

    return aten__log_softmax(self, dim, half_to_float)


@torch_op("aten::_log_softmax")
def aten__log_softmax(
    self: TFloatHighPrecision, dim: int, half_to_float: bool  # pylint: disable=unused-argument
) -> TFloatHighPrecision:
    """_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""

    self_is_scalar = op.Size(op.Shape(self)) == 0
    if self_is_scalar:
        self = op.Unsqueeze(self, op.Constant(value_ints=[0]))
    result = op.LogSoftmax(self, axis=dim)
    if self_is_scalar:  # squeeze to scalar due to input is scalar
        result = op.Squeeze(result)
    return result


@torch_op("aten::_softmax", trace_only=True)
def aten__softmax_half(self: Union[FLOAT16, BFLOAT16], dim: int, half_to_float: bool) -> FLOAT:
    """_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""

    # trace_only because we need to cast conditionally based on half_to_float
    if half_to_float:
        self = op.Cast(self, to=FLOAT.dtype)

    return aten_softmax_no_dtype(self, dim)


@torch_op("aten::_softmax", trace_only=True)
def aten__softmax(
    self: TFloatHighPrecision, dim: int, half_to_float: bool
) -> TFloatHighPrecision:
    """_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""

    # trace_only to reuse aten_softmax_no_dtype

    del half_to_float  # Unused
    return aten_softmax_no_dtype(self, dim)


@torch_op("aten::abs")
def aten_abs(self: TRealOrUInt8) -> TRealOrUInt8:
    """abs(Tensor self) -> Tensor"""

    return op.Abs(self)


@torch_op("aten::abs", complex=True)
def aten_abs_complex(self: TRealOrUInt8) -> TRealOrUInt8:
    """abs(Tensor self) -> Tensor"""
    # self_real = self[..., 0]
    self_real = op.Gather(self, 0, axis=-1)
    # self_imag = self[..., 1]
    self_imag = op.Gather(self, 1, axis=-1)
    real_pow = op.Pow(self_real, 2)
    imag_pow = op.Pow(self_imag, 2)
    real_plus_imag = op.Add(real_pow, imag_pow)
    return op.Sqrt(real_plus_imag)


@torch_op("aten::acos")
def aten_acos(self: TFloat) -> TFloat:
    """acos(Tensor self) -> Tensor"""

    return op.Acos(self)


@torch_op("aten::acosh")
def aten_acosh(self: TFloat) -> TFloat:
    """acosh(Tensor self) -> Tensor"""

    return op.Acosh(self)


@torch_op(("aten::add", "aten::add.Tensor"))
def aten_add(self: TReal, other: TReal, alpha: float = 1.0) -> TReal:
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    # TODO(microsoft/onnxruntime#15977): Improve fp16 precision
    alpha = op.CastLike(alpha, other)
    other = op.Mul(other, alpha)
    return op.Add(self, other)


@torch_op("aten::addbmm")
def aten_addbmm(
    self: TReal,
    batch1: TReal,
    batch2: TReal,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> TReal:
    """addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

    Performs a batch matrix-matrix product of matrices stored in `batch1` and `batch2`,
    with a reduced add step (all matrix multiplications get accumulated along the first
    dimension). `self` is added to the final result.

    `batch1` and `batch2` must be 3-D tensors each containing the same number of matrices.
    """

    scaled_self = op.Mul(self, beta)
    axes = op.Constant(value_ints=[0])
    reduced_batches = op.ReduceSum(op.MatMul(batch1, batch2), axes, keepdims=False)

    return op.Add(scaled_self, op.Mul(reduced_batches, alpha))


@torch_op("aten::addcdiv")
def aten_addcdiv(self: TFloat, tensor1: TFloat, tensor2: TFloat, value: float = 1.0) -> TFloat:
    """addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

    Performs the element-wise division of tensor1 by tensor2, multiplies the result
    by the scalar value and adds it to self.
    """

    return op.Add(self, op.Mul(op.Div(tensor1, tensor2), value))


@torch_op("aten::addcmul")
def aten_addcmul(
    self: TReal,
    tensor1: TReal,
    tensor2: TReal,
    value: float = 1.0,
) -> TReal:
    """addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

    Performs the element-wise multiplication of tensor1 by tensor2, multiplies the
    result by the scalar value and adds it to self.
    """

    # Follow the order in https://github.com/pytorch/pytorch/blob/29e3fddb082b5a14262a7246bc62381a55199d45/aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp#L47
    # TODO(#811): Understand fp16 accuracy issue
    return op.Add(self, op.Mul(op.Mul(value, tensor1), tensor2))


@torch_op("aten::addmm")
def aten_addmm(
    self: TReal, mat1: TReal, mat2: TReal, beta: float = 1.0, alpha: float = 1.0
) -> TReal:
    """addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""

    # NOTE: ONNX Runtime does not support int inputs to Gemm as of 1.16.
    # To support int inputs, consider an overriding implementation that casts to float and back.

    # addmm only accepts 2d tensors: https://pytorch.org/docs/stable/generated/torch.addmm.html
    return op.Gemm(mat1, mat2, self, alpha=alpha, beta=beta)


@torch_op("aten::addmv")
def aten_addmv(
    self: TReal, mat: TReal, vec: TReal, beta: float = 1.0, alpha: float = 1.0
) -> TReal:
    """addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""

    return op.Add(op.Mul(self, beta), op.Mul(op.MatMul(mat, vec), alpha))


@torch_op("aten::addr")
def aten_addr(
    self: TReal, vec1: TReal, vec2: TReal, beta: float = 1.0, alpha: float = 1.0
) -> TReal:
    """addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

    Performs the outer-product of vectors vec1 and vec2 and adds it to the matrix input.
    """
    vec1_shape = op.Constant(value_ints=[-1, 1])
    vec2_shape = op.Constant(value_ints=[1, -1])
    vec1_reshaped = op.Reshape(vec1, vec1_shape)
    vec2_reshaped = op.Reshape(vec2, vec2_shape)

    outer = op.MatMul(vec1_reshaped, vec2_reshaped)
    # https://github.com/pytorch/pytorch/blob/51664489ba6f6b2343bbec9af9ca99185e2a5dbc/aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp#L53-L54
    # When beta == 0, values in self should be ignored,
    # nans and infs in self should not propagate.
    if beta == 0.0:
        result = op.Mul(alpha, outer)
    else:
        result = op.Add(op.Mul(beta, self), op.Mul(alpha, outer))

    return result


def aten_adjoint(self: TensorType) -> TensorType:
    """adjoint(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


def aten_affine_grid_generator(
    theta: TensorType, size: Sequence[int], align_corners: bool
) -> TensorType:
    """affine_grid_generator(Tensor theta, int[] size, bool align_corners) -> Tensor"""

    raise NotImplementedError()


def aten_affine_grid_generator_backward(
    grad: TensorType, size: Sequence[int], align_corners: bool
) -> TensorType:
    """affine_grid_generator_backward(Tensor grad, int[] size, bool align_corners) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::alias")
def aten_alias(self: TTensor) -> TTensor:
    """alias(Tensor(a) self) -> Tensor(a)"""

    return op.Identity(self)


def aten_alias_copy(self: TensorType) -> TensorType:
    """alias_copy(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_align_as(self: TensorType, other: TensorType) -> TensorType:
    """align_as(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_align_tensors(tensors: Sequence[TensorType]) -> TensorType:
    """align_tensors(Tensor[] tensors) -> Tensor[]"""

    raise NotImplementedError()


def aten_align_to(self: TensorType, names: Sequence[str]) -> TensorType:
    """align_to(Tensor(a) self, Dimname[] names) -> Tensor(a)"""

    raise NotImplementedError()


@torch_op("aten::all")
def aten_all(self: TTensor) -> BOOL:
    """all(Tensor self) -> Tensor"""

    if IsScalar(self):
        result = op.Cast(self, to=BOOL.dtype)
    else:
        self_bool = op.Cast(self, to=BOOL.dtype)
        self_int = op.Cast(self_bool, to=INT64.dtype)
        all_true = op.ReduceMin(self_int, keepdims=False)
        result = op.Cast(all_true, to=BOOL.dtype)
    return result


@torch_op("aten::all.dim")
def aten_all_dim(self: TTensor, dim: int, keepdim: bool = False) -> BOOL:
    """all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"""

    self_rank = op.Size(op.Shape(self))
    if self_rank == 0:
        result = op.Cast(self, to=BOOL.dtype)
    else:
        self_bool = op.Cast(self, to=BOOL.dtype)
        self_int = op.Cast(self_bool, to=INT64.dtype)
        dims = op.Reshape(dim, op.Constant(value_ints=[-1]))
        all_true = op.ReduceMin(self_int, dims, keepdims=keepdim)
        result = op.Cast(all_true, to=BOOL.dtype)
    return result


@torch_op("aten::all.dims", trace_only=True)
def aten_all_dims(self: TTensor, dim: Sequence[int] = (), keepdim: bool = False) -> BOOL:
    """all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor"""

    if not dim:
        return aten_all_dims_no_dim(self, keepdim)
    for d in dim:
        self = aten_all_dim(self, d, keepdim)
    return self


@torch_op("aten::all.dims")
def aten_all_dims_no_dim(self: TTensor, keepdims: bool) -> BOOL:
    """all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor"""

    # dim is None and thus not supplied

    self_rank = op.Size(op.Shape(self))
    if self_rank == 0:
        result = op.Cast(self, to=BOOL.dtype)
    else:
        self_bool = op.Cast(self, to=BOOL.dtype)
        self_int = op.Cast(self_bool, to=INT64.dtype)
        all_true = op.ReduceMin(self_int, keepdims=keepdims)
        result = op.Cast(all_true, to=BOOL.dtype)
    return result


@torch_op("aten::allclose")
def aten_allclose(
    self: TReal,
    other: TReal,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,  # pylint: disable=unused-argument
) -> BOOL:
    """allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool"""

    # FIXME: check equal_nan when self and other are all NaN
    # |input - other| <= atol + rtol x |other|
    left_part = op.Abs(op.Sub(self, other))
    right_part = op.Add(atol, op.Mul(rtol, op.Abs(other)))
    is_close = op.LessOrEqual(left_part, right_part)
    is_close_int = op.Cast(is_close, to=INT8.dtype)

    # If min is 0, some elements are not close -> allclose is False
    # If min is 1, all elements are close -> allclose is True
    return op.Cast(op.ReduceMin(is_close_int, keepdims=False), to=BOOL.dtype)


def aten_alpha_dropout(input: TensorType, p: float, train: bool) -> TensorType:
    """alpha_dropout(Tensor input, float p, bool train) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::amax")
def aten_amax(self: TRealOrUInt8, dim: INT64, keepdim: bool = False) -> TRealOrUInt8:
    """amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"""

    # ReduceMax reduces all dimensions when dim is empty
    return op.ReduceMax(self, dim, keepdims=keepdim)


@torch_op("aten::amin")
def aten_amin(self: TRealOrUInt8, dim: INT64, keepdim: bool = False) -> TRealOrUInt8:
    """amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"""

    # ReduceMin reduces all dimensions when dim is empty
    return op.ReduceMin(self, dim, keepdims=keepdim)


def aten_aminmax(
    self: TensorType, dim: Optional[int] = None, keepdim: bool = False
) -> tuple[TensorType, TensorType]:
    """aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)"""

    raise NotImplementedError()


def aten_and(self: TensorType, other: TensorType) -> TensorType:
    """__and__.Tensor(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_angle(self: TensorType) -> TensorType:
    """angle(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::any")
def aten_any(self: TTensor) -> BOOL:
    """any(Tensor self) -> Tensor"""

    self_rank = op.Size(op.Shape(self))
    if self_rank == 0:
        result = op.Cast(self, to=BOOL.dtype)
    else:
        self_bool = op.Cast(self, to=BOOL.dtype)
        # op.ReduceMax() in the next step cannot process BOOL inputs, so convert to INT64
        self_int = op.Cast(self_bool, to=INT64.dtype)
        any_true = op.ReduceMax(self_int, keepdims=False)
        result = op.Cast(any_true, to=BOOL.dtype)
    return result


@torch_op("aten::any.dim")
def aten_any_dim(self: TTensor, dim: int, keepdim: bool = False) -> BOOL:
    """any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"""

    self_rank = op.Size(op.Shape(self))
    if self_rank == 0:
        result = op.Cast(self, to=BOOL.dtype)
    else:
        self_bool = op.Cast(self, to=BOOL.dtype)
        # op.ReduceMax() in the next step cannot process BOOL inputs, so convert to INT64
        self_int = op.Cast(self_bool, to=INT64.dtype)
        # Change dim from int to INT64[1]
        dims = op.Reshape(dim, op.Constant(value_ints=[-1]))
        any_true = op.ReduceMax(self_int, dims, keepdims=keepdim)
        result = op.Cast(any_true, to=BOOL.dtype)
    return result


@torch_op("aten::any.dims", trace_only=True)
def aten_any_dims(self: TTensor, dim: Sequence[int] = (), keepdim: bool = False) -> BOOL:
    """any.dims(Tensor self, int[1]? dim=None, bool keepdim=False) -> Tensor"""

    if not dim:
        return aten_any_dims_no_dim(self, keepdim)
    for d in dim:
        self = aten_any_dim(self, d, keepdim)
    return self


@torch_op("aten::any.dims")
def aten_any_dims_no_dim(self: TTensor, keepdims: bool) -> BOOL:
    """any.dims(Tensor self, int[1]? dim=None, bool keepdim=False) -> Tensor"""

    # dim is None and thus not supplied

    self_rank = op.Size(op.Shape(self))
    if self_rank == 0:
        result = op.Cast(self, to=BOOL.dtype)
    else:
        self_bool = op.Cast(self, to=BOOL.dtype)
        self_int = op.Cast(self_bool, to=INT64.dtype)
        any_true = op.ReduceMax(self_int, keepdims=keepdims)
        result = op.Cast(any_true, to=BOOL.dtype)
    return result


def _range_supported(dtype: int) -> bool:
    """Returns true if the dtype is supported by the ONNX Range op."""
    return dtype in {
        DOUBLE.dtype,
        FLOAT.dtype,
        INT16.dtype,
        INT32.dtype,
        INT64.dtype,
    }


def _integral_to_be_adjusted(dtype: int) -> bool:
    """Returns true if the dtype is special integral handled by torch."""
    return dtype in {
        INT8.dtype,
        INT16.dtype,
        INT32.dtype,
    }


@torch_op("aten::arange", trace_only=True)
def aten_arange(end: Union[DOUBLE, FLOAT, INT16, INT32, INT64], dtype: int = -1) -> TensorType:
    """arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    # NOTE: trace_only because both if branches need to be the same type, but we have
    # a cast in the if branch.

    if dtype == -1:
        zero = op.CastLike(0.0, end)
        one = op.CastLike(1.0, end)
        result = op.Range(zero, end, one)
    elif _range_supported(dtype):
        end = op.Cast(end, to=dtype)
        zero = op.Cast(0, to=dtype)
        one = op.Cast(1, to=dtype)
        result = op.Range(zero, end, one)
    else:
        # Cast input to float if dtype is not supported by Range,
        # because the input dtype may be e.g. bfloat16 / int8 etc.
        # which Range does not support. The output type is ensured because the output
        # is casted to the specified dtype.
        end = op.Cast(end, to=FLOAT.dtype)
        zero = op.Constant(value_float=0.0)
        one = op.Constant(value_float=1.0)
        result = op.Cast(op.Range(zero, end, one), to=dtype)

    return result


@torch_op("aten::arange.start", trace_only=True)
def aten_arange_start(
    start: TRealUnlessFloat16OrInt8, end: TRealUnlessFloat16OrInt8, dtype: int = -1
) -> TensorType:
    """arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    # NOTE: trace_only because both if branches need to be the same type, but we have
    # a cast in the if branch.

    if dtype == -1:
        one = op.CastLike(1.0, end)
        result = op.Range(start, end, one)
    elif _range_supported(dtype):
        end = op.Cast(end, to=dtype)
        start = op.Cast(start, to=dtype)
        one = op.Cast(1, to=dtype)
        result = op.Range(start, end, one)
    else:
        # Cast input to float if dtype is not supported by Range,
        # because the input dtype may be e.g. bfloat16 / int8 etc.
        # which Range does not support. The output type is ensured because the output
        # is casted to the specified dtype.
        end = op.Cast(end, to=FLOAT.dtype)
        start = op.Cast(start, to=FLOAT.dtype)
        one = op.Constant(value_float=1.0)
        result = op.Cast(op.Range(start, end, one), to=dtype)

    return result


@torch_op("aten::arange.start_step", private=True)
def _adjust_args_for_arange_int_dtype(
    start: TRealUnlessFloat16OrInt8,
    end: TRealUnlessFloat16OrInt8,
    step: TRealUnlessFloat16OrInt8,
) -> Tuple[FLOAT, FLOAT, FLOAT]:
    zero = op.Cast(0.0, to=FLOAT.dtype)
    start = op.Cast(start, to=FLOAT.dtype)
    end = op.Cast(end, to=FLOAT.dtype)
    step = op.Cast(step, to=FLOAT.dtype)

    start = op.Where(op.Less(start, zero), op.Ceil(start), start)
    start = op.Where(op.Less(step, zero), op.Floor(start), start)

    return (start, end, step)


@torch_op("aten::arange.start_step", trace_only=True)
def aten_arange_start_step(
    start: TRealUnlessFloat16OrInt8,
    end: TRealUnlessFloat16OrInt8,
    step: TRealUnlessFloat16OrInt8,
    dtype: int = -1,
) -> TensorType:
    """arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    # NOTE: trace_only because both if branches need to be the same type, but we have
    # a cast in the if branch.

    if dtype == -1:
        result = op.Range(start, end, step)
    elif _integral_to_be_adjusted(dtype):
        # PyTorch arange op handles these integral types differently from INT64,
        # so we have to adjust these arguments accordingly.
        # https://github.com/pytorch/pytorch/blob/121cfb60c0817816fcbe2190303b7f6d05c77cf3/torch/_refs/__init__.py#L4794
        start, end, step = _adjust_args_for_arange_int_dtype(start, end, step)
        result = op.Cast(op.Range(start, end, step), to=dtype)
    elif dtype == INT64.dtype:
        end = op.Cast(end, to=dtype)
        start = op.Cast(start, to=dtype)
        step = op.Cast(step, to=dtype)
        result = op.Range(start, end, step)
    else:
        # Cast input to float if dtype is not supported by Range,
        # because the input dtype may be e.g. bfloat16,
        # which Range does not support. The output type is ensured because the output
        # is casted to the specified dtype.
        end = op.Cast(end, to=FLOAT.dtype)
        start = op.Cast(start, to=FLOAT.dtype)
        step = op.Cast(step, to=FLOAT.dtype)
        result = op.Cast(op.Range(start, end, step), to=dtype)

    return result


def aten_arccos(self: TensorType) -> TensorType:
    """arccos(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_arccosh(self: TensorType) -> TensorType:
    """arccosh(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_arcsin(self: TensorType) -> TensorType:
    """arcsin(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_arcsinh(self: TensorType) -> TensorType:
    """arcsinh(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_arctan(self: TensorType) -> TensorType:
    """arctan(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_arctan2(self: TensorType, other: TensorType) -> TensorType:
    """arctan2(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_arctanh(self: TensorType) -> TensorType:
    """arctanh(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::argmax")
def aten_argmax(self: Union[RealType, UINT8], keepdim: bool = False) -> INT64:
    """argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""

    self_is_scaler = op.Size(op.Shape(self)) == 0
    self = op.Reshape(self, op.Constant(value_ints=[-1]))
    result = op.ArgMax(self, keepdims=keepdim)
    if self_is_scaler:
        result = op.Squeeze(result)

    return result


@torch_op("aten::argmax")
def aten_argmax_dim(self: Union[RealType, UINT8], dim: int, keepdim: bool = False) -> INT64:
    """argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""

    self_is_scaler = op.Size(op.Shape(self)) == 0
    if self_is_scaler:
        self = op.Reshape(self, op.Constant(value_ints=[-1]))

    result = op.ArgMax(self, axis=dim, keepdims=keepdim)
    if self_is_scaler:
        result = op.Squeeze(result)

    return result


@torch_op("aten::argmin")
def aten_argmin(self: Union[RealType, UINT8], keepdim: bool = False) -> INT64:
    """argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""

    self_is_scaler = op.Size(op.Shape(self)) == 0
    self = op.Reshape(self, op.Constant(value_ints=[-1]))
    result = op.ArgMin(self, keepdims=keepdim)
    if self_is_scaler:
        result = op.Squeeze(result)

    return result


@torch_op("aten::argmin")
def aten_argmin_dim(self: Union[RealType, UINT8], dim: int, keepdim: bool = False) -> INT64:
    """argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""

    self_is_scaler = op.Size(op.Shape(self)) == 0
    if self_is_scaler:
        self = op.Reshape(self, op.Constant(value_ints=[-1]))

    result = op.ArgMin(self, axis=dim, keepdims=keepdim)
    if self_is_scaler:
        result = op.Squeeze(result)

    return result


def aten_argsort(self: TensorType, dim: int = -1, descending: bool = False) -> TensorType:
    """argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor"""

    raise NotImplementedError()


def aten_argwhere(self: TensorType) -> TensorType:
    """argwhere(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::as_strided", trace_only=True)
def aten_as_strided(
    self: TTensor, size: INT64, stride: INT64, storage_offset: int = 0
) -> TTensor:
    """as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)"""

    rank = len(stride)
    return _aten_as_strided_onnx(self, size, stride, storage_offset, rank)


@torch_op("aten::as_strided", private=True)
def _aten_as_strided_onnx(
    self: TTensor, size: INT64, stride: INT64, storage_offset: int = 0, rank: int = 0
) -> TTensor:
    # e.g. when size=[2,3,4], stride=[2,1,3], indices=[0]
    # i = 0
    # indices=[0], add_value=[0,3,6,9]
    # expand(shape=[4]) to [0,0,0,0]
    # then + add_value = [0,3,6,9]
    # i = 1
    # indices=[0,3,6,9], add_value=[0,1,2]
    # expand(shape=[3,4] to [[0,3,6,9],[0,3,6,9],[0,3,6,9]]
    # indices + add_value = [[0,3,6,9],[1,3,7,10],[2,5,8,11]]
    # i = 2
    # indices = [[0,3,6,9],[1,3,7,10],[2,5,8,11]], add_value=[0,2]
    # expand(shape=[2,3,4]) to [[[0,3,6,9],[1,3,7,10],[2,5,8,11]]],[[0,3,6,9],[1,3,7,10],[2,5,8,11]]]
    # indices + add_value = [[[0,3,6,9],[1,3,7,10],[2,5,8,11]]],[[2,5,8,11],[3,5,9,12],[4,7,10,13]]]
    neg_1 = op.Constant(value_ints=[-1])
    rank_tensor = op.Reshape(rank, neg_1)  # should be 3
    # The final indices for op.Gather(data, indices), will be continually changed during the loop
    indices = op.Constant(value_int=0)
    one_seq = op.SequenceEmpty()
    for i in range(rank):
        # Get the index from back to front, should be 2,1,0 when to i=0,1,2
        j = rank - i - 1
        j_tensor = op.Reshape(j, neg_1)
        # Get size according to index_j, should be 4,3,2 when i=0,1,2
        size_dim_j = op.Gather(size, j_tensor, axis=0)
        # Get right size according to index_j, should be [4],[3,4],[2,3,4] when i=0,1,2
        size_after_j = op.Slice(size, j_tensor, rank_tensor)
        # Get stride according to index_j, should be 3,1,2 when i=0,1,2
        stride_dim_j = op.Gather(stride, j_tensor, axis=0)
        indices = op.Expand(indices, size_after_j)
        # When size[j]=4, stride[j]=3, then add_value = [0,1,2,3] * 3 = [0,3,6,9]
        # When size[j]=3, stride[j]=1, then add_value = [0,1,2] * 1 = [0,1,2]
        # When size[j]=2, stride[j]=2, then add_value = [0,1] * 2 = [0,2]
        add_value = op.Range(0, size_dim_j, 1) * stride_dim_j
        # Compute the shape for add_value for correct broadcasting
        if i == 0:
            # shape = [dim_size]
            shape = size_dim_j
        else:
            # shape = [dim_size, 1, 1, ...], the count of 1 euqal to i
            ones = op.ConcatFromSequence(one_seq, axis=0)
            shape = op.Concat(op.Cast(size_dim_j, to=FLOAT.dtype), ones, axis=0)
            shape = op.Cast(shape, to=INT64.dtype)

        add_value = op.Reshape(add_value, shape)
        # Broadcasting add value to indices according to size and stride value
        indices = indices + add_value
        # Dims after dim_size to reshape(add_value), should be [1],[1,1],[1,1,1] when i=0,1,2
        one_seq = op.SequenceInsert(one_seq, op.Constant(value_floats=[1.0]))

    self_flatten = op.Reshape(self, op.Constant(value_ints=[-1]))
    indices = op.Add(indices, storage_offset)
    result = op.Gather(self_flatten, indices)

    return result


def aten_as_strided_copy(
    self: TensorType, size: INT64, stride: INT64, storage_offset: Optional[INT64] = None
) -> TensorType:
    """as_strided_copy(Tensor self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor"""

    raise NotImplementedError()


def aten_as_strided_scatter(
    self: TensorType,
    src: TensorType,
    size: INT64,
    stride: INT64,
    storage_offset: Optional[INT64] = None,
) -> TensorType:
    """as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::asin")
def aten_asin(self: TFloat) -> TFloat:
    """asin(Tensor self) -> Tensor"""

    return op.Asin(self)


@torch_op("aten::asinh")
def aten_asinh(self: TFloat) -> TFloat:
    """asinh(Tensor self) -> Tensor"""

    return op.Asinh(self)


@torch_op("aten::atan")
def aten_atan(self: TFloat) -> TFloat:
    """atan(Tensor self) -> Tensor"""

    return op.Atan(self)


@torch_op("aten::atan2")
def aten_atan2(self: TFloat, other: TFloat) -> TFloat:
    """atan2(Tensor self, Tensor other) -> Tensor"""

    # self is y, and other is x on coordinate
    slope = op.Div(self, other)
    atan = op.Atan(slope)

    second_third_quadrant = op.Where(self > 0.0, atan + _MATH_PI, atan - _MATH_PI)
    result = op.Where(other < 0.0, second_third_quadrant, atan)

    return result


@torch_op("aten::atanh")
def aten_atanh(self: TFloat) -> TFloat:
    """atanh(Tensor self) -> Tensor"""

    return op.Atanh(self)


@torch_op("aten::atleast_1d")
def aten_atleast_1d(self: TTensor) -> TTensor:
    """atleast_1d(Tensor self) -> Tensor"""

    shape = op.Shape(self)
    rank = op.Size(shape)
    if rank == 0:
        self = op.Reshape(self, op.Constant(value_ints=[1]))
    return self


@torch_op("aten::atleast_1d.Sequence")
def aten_atleast_1d_sequence(self: Sequence[TTensor]) -> TTensor:
    """atleast_1d.Sequence(Tensor[] tensors) -> Tensor[]"""

    @graph()
    def reshape_to_1d(tensor):
        shape = op.Shape(tensor)
        rank = op.Size(shape)
        if rank == 0:
            tensor = op.Reshape(tensor, op.Constant(value_ints=[1]))
        return tensor

    return op.SequenceMap(self, body=reshape_to_1d)


@torch_op("aten::atleast_2d")
def aten_atleast_2d(self: TTensor) -> TTensor:
    """atleast_2d(Tensor self) -> Tensor"""

    shape = op.Shape(self)
    rank = op.Size(shape)
    if rank <= 1:
        self = op.Reshape(self, op.Constant(value_ints=[1, -1]))
    return self


@torch_op("aten::atleast_2d.Sequence")
def aten_atleast_2d_sequence(self: Sequence[TTensor]) -> TTensor:
    """atleast_2d.Sequence(Tensor[] tensors) -> Tensor[]"""

    @graph()
    def reshape_to_2d(tensor):
        shape = op.Shape(tensor)
        rank = op.Size(shape)
        if rank <= 1:
            tensor = op.Reshape(tensor, op.Constant(value_ints=[1, -1]))
        return tensor

    return op.SequenceMap(self, body=reshape_to_2d)


@torch_op("aten::atleast_3d")
def aten_atleast_3d(self: TTensor) -> TTensor:
    """atleast_3d(Tensor self) -> Tensor"""

    shape = op.Shape(self)
    rank = op.Size(shape)
    if rank <= 1:
        self = op.Reshape(self, op.Constant(value_ints=[1, -1, 1]))
    elif rank == 2:
        self = op.Unsqueeze(self, op.Constant(value_ints=[-1]))
    return self


@torch_op("aten::atleast_3d.Sequence")
def aten_atleast_3d_sequence(self: Sequence[TTensor]) -> TTensor:
    """atleast_3d.Sequence(Tensor[] tensors) -> Tensor[]"""

    @graph()
    def reshape_to_3d(tensor):
        shape = op.Shape(tensor)
        rank = op.Size(shape)
        if rank <= 1:
            tensor = op.Reshape(tensor, op.Constant(value_ints=[1, -1, 1]))
        elif rank == 2:
            tensor = op.Unsqueeze(tensor, op.Constant(value_ints=[-1]))
        return tensor

    return op.SequenceMap(self, body=reshape_to_3d)


@torch_op("aten::baddbmm")
def aten_baddbmm(
    self: TRealOrUInt8,
    batch1: TRealUnlessInt16OrInt8,
    batch2: TRealUnlessInt16OrInt8,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> TRealUnlessInt16OrInt8:
    """baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""
    batch_mul = op.MatMul(batch1, batch2)
    alpha_cast = op.CastLike(alpha, self)
    mul_a = op.Mul(batch_mul, alpha_cast)
    beta_cast = op.CastLike(beta, self)
    mul_b = op.Mul(self, beta_cast)
    return op.Add(mul_a, mul_b)


def aten_bartlett_window(window_length: int) -> TensorType:
    """bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

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
    """batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor"""

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
    """batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu, Tensor count) -> Tensor"""

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
    """batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)"""

    raise NotImplementedError()


def aten_batch_norm_elemt(
    input: TensorType,
    weight: Optional[TensorType],
    bias: Optional[TensorType],
    mean: TensorType,
    invstd: TensorType,
    eps: float,
) -> TensorType:
    """batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor"""

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
    """batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)"""

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
    """batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_batch_norm_stats(input: TensorType, eps: float) -> tuple[TensorType, TensorType]:
    """batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_batch_norm_update_stats(
    input: TensorType,
    running_mean: Optional[TensorType],
    running_var: Optional[TensorType],
    momentum: float,
) -> tuple[TensorType, TensorType]:
    """batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)"""

    raise NotImplementedError()


@torch_op("aten::bernoulli")
def aten_bernoulli(self: TFloat) -> TFloat:
    """Proximal implementation of aten::bernoulli.default

    Note that due to the limitation of ONNX, we ignore the `generator` argument in
      aten::bernoulli.default(Tensor self, *, Generator? generator=None) -> Tensor
    """
    return op.Bernoulli(self)


@torch_op("aten::bernoulli.p")
def aten_bernoulli_p(self: TTensor, p: float) -> TTensor:
    """Proximal implementation of aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None)

    Ignore `generator` due to the limit on ONNX expressiveness.
    """
    # NOTE: We will lose some precision when input is float64 but that's considered insignificant
    self_float = op.Cast(self, to=FLOAT.dtype)
    rands = op.RandomUniformLike(
        self_float,
        high=1.0,
        low=0.0,
    )
    sampled = op.Less(rands, p)
    return op.CastLike(sampled, self)


def aten_bilinear(
    input1: TensorType,
    input2: TensorType,
    weight: TensorType,
    bias: Optional[TensorType] = None,
) -> TensorType:
    """bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None) -> Tensor"""

    raise NotImplementedError()


def aten_binary_cross_entropy_with_logits(
    self: TensorType,
    target: TensorType,
    weight: Optional[TensorType] = None,
    pos_weight: Optional[TensorType] = None,
    reduction: int = 1,
) -> TensorType:
    """binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


def aten_bincount(
    self: TensorType, weights: Optional[TensorType] = None, minlength: int = 0
) -> TensorType:
    """bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor"""

    raise NotImplementedError()


def aten_binomial(
    count: TensorType, prob: TensorType, generator: Optional[str] = None
) -> TensorType:
    """binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor"""

    raise NotImplementedError()


@torch_op(
    (
        "aten::bitwise_and",
        "aten::bitwise_and.Tensor",
        "aten::bitwise_and.Scalar",
        "aten::bitwise_and.Scalar_Tensor",
    )
)
def aten_bitwise_and(self: TInt, other: TInt) -> TInt:
    """bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor"""
    # logical_and implements the BOOL variant

    return op.BitwiseAnd(self, other)


@torch_op("aten::bitwise_left_shift")
def aten_bitwise_left_shift_int16(self: INT16, other: INT16) -> INT16:
    """bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    # assert other >= 0
    self = op.Cast(self, to=UINT16.dtype)
    other = op.Cast(other, to=UINT16.dtype)

    result = op.BitShift(self, other, direction="LEFT")

    return op.Cast(result, to=INT16.dtype)


@torch_op("aten::bitwise_left_shift")
def aten_bitwise_left_shift_int32(self: INT32, other: INT32) -> INT32:
    """bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    # assert other >= 0
    self = op.Cast(self, to=UINT32.dtype)
    other = op.Cast(other, to=UINT32.dtype)

    result = op.BitShift(self, other, direction="LEFT")

    return op.Cast(result, to=INT32.dtype)


@torch_op("aten::bitwise_left_shift")
def aten_bitwise_left_shift_int64(self: INT64, other: INT64) -> INT64:
    """bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    # assert other >= 0
    self = op.Cast(self, to=UINT64.dtype)
    other = op.Cast(other, to=UINT64.dtype)

    result = op.BitShift(self, other, direction="LEFT")

    return op.Cast(result, to=INT64.dtype)


@torch_op("aten::bitwise_left_shift")
def aten_bitwise_left_shift_int8(self: INT8, other: INT8) -> INT8:
    """bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    # assert other >= 0
    self = op.Cast(self, to=UINT8.dtype)
    other = op.Cast(other, to=UINT8.dtype)

    result = op.BitShift(self, other, direction="LEFT")

    return op.Cast(result, to=INT8.dtype)


@torch_op("aten::bitwise_not")
def aten_bitwise_not(self: TInt) -> TInt:
    """bitwise_not(Tensor self) -> Tensor"""
    # logical_not implements the BOOL variant

    return op.BitwiseNot(self)


@torch_op(
    (
        "aten::bitwise_or",
        "aten::bitwise_or.Tensor",
        "aten::bitwise_or.Scalar",
        "aten::bitwise_or.Scalar_Tensor",
    )
)
def aten_bitwise_or(self: TInt, other: TInt) -> TInt:
    """bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor"""
    # logical_or implements the BOOL variant

    return op.BitwiseOr(self, other)


@torch_op("aten::bitwise_right_shift")
def aten_bitwise_right_shift_int16(self: INT16, other: INT16) -> INT16:
    """bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    negative = op.Less(self, 0)
    self = op.Cast(self, to=UINT16.dtype)
    other = op.Cast(other, to=UINT16.dtype)

    # Simulate arithmetic shift using logical shift
    # Clear the lower bits of an all one mask to create the mask to simulate the sign bit shifting
    mask = op.BitShift(
        op.Cast(op.Constant(value_int=0xFFFF), to=UINT16.dtype), other, direction="RIGHT"
    )
    mask = op.BitwiseNot(mask)
    # Do logical shift
    shifted = op.BitShift(self, other, direction="RIGHT")
    # Compute the arithmetic shifted value assuming the sign bit was set
    negative_shifted = op.BitwiseOr(shifted, mask)
    # Choose the shifted value based on the sign bit
    return op.Where(
        negative, op.Cast(negative_shifted, to=INT16.dtype), op.Cast(shifted, to=INT16.dtype)
    )


@torch_op("aten::bitwise_right_shift")
def aten_bitwise_right_shift_int32(self: INT32, other: INT32) -> INT32:
    """bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    negative = op.Less(self, 0)
    self = op.Cast(self, to=UINT32.dtype)
    other = op.Cast(other, to=UINT32.dtype)

    # Simulate arithmetic shift using logical shift
    # Clear the lower bits of an all one mask to create the mask to simulate the sign bit shifting
    mask = op.BitShift(
        op.Cast(op.Constant(value_int=0xFFFFFFFF), to=UINT32.dtype), other, direction="RIGHT"
    )
    mask = op.BitwiseNot(mask)
    # Do logical shift
    shifted = op.BitShift(self, other, direction="RIGHT")
    # Compute the arithmetic shifted value assuming the sign bit was set
    negative_shifted = op.BitwiseOr(shifted, mask)
    # Choose the shifted value based on the sign bit
    return op.Where(
        negative, op.Cast(negative_shifted, to=INT32.dtype), op.Cast(shifted, to=INT32.dtype)
    )


@torch_op("aten::bitwise_right_shift")
def aten_bitwise_right_shift_int64(self: INT64, other: INT64) -> INT64:
    """bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    negative = op.Less(self, 0)
    self = op.Cast(self, to=UINT64.dtype)
    other = op.Cast(other, to=UINT64.dtype)

    # Simulate arithmetic shift using logical shift
    # Clear the lower bits of an all one mask to create the mask to simulate the sign bit shifting
    mask = op.BitShift(
        # 0xFFFFFFFFFFFFFFFF
        op.Cast(op.Constant(value_int=-1), to=UINT64.dtype),
        other,
        direction="RIGHT",
    )
    mask = op.BitwiseNot(mask)
    # Do logical shift
    shifted = op.BitShift(self, other, direction="RIGHT")
    # Compute the arithmetic shifted value assuming the sign bit was set
    negative_shifted = op.BitwiseOr(shifted, mask)
    # Choose the shifted value based on the sign bit
    return op.Where(
        negative, op.Cast(negative_shifted, to=INT64.dtype), op.Cast(shifted, to=INT64.dtype)
    )


@torch_op("aten::bitwise_right_shift")
def aten_bitwise_right_shift_int8(self: INT8, other: INT8) -> INT8:
    """bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    negative = op.Less(self, 0)
    self = op.Cast(self, to=UINT8.dtype)
    other = op.Cast(other, to=UINT8.dtype)

    # Simulate arithmetic shift using logical shift
    # Clear the lower bits of an all one mask to create the mask to simulate the sign bit shifting
    mask = op.BitShift(
        op.Cast(op.Constant(value_int=0xFF), to=UINT8.dtype), other, direction="RIGHT"
    )
    mask = op.BitwiseNot(mask)
    # Do logical shift
    shifted = op.BitShift(self, other, direction="RIGHT")
    # Compute the arithmetic shifted value assuming the sign bit was set
    negative_shifted = op.BitwiseOr(shifted, mask)
    # Choose the shifted value based on the sign bit
    return op.Where(
        negative, op.Cast(negative_shifted, to=INT8.dtype), op.Cast(shifted, to=INT8.dtype)
    )


@torch_op(
    (
        "aten::bitwise_xor",
        "aten::bitwise_xor.Tensor",
        "aten::bitwise_xor.Scalar",
        "aten::bitwise_xor.Scalar_Tensor",
    )
)
def aten_bitwise_xor(self: TInt, other: TInt) -> TInt:
    """bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor"""
    # logical_xor implements the BOOL variant

    return op.BitwiseXor(self, other)


def aten_blackman_window(window_length: int) -> TensorType:
    """blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


def aten_block_diag(tensors: Sequence[TensorType]) -> TensorType:
    """block_diag(Tensor[] tensors) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::bmm")
def aten_bmm(self: TFloat, mat2: TFloat) -> TFloat:
    """bmm(Tensor self, Tensor mat2) -> Tensor"""

    return op.MatMul(self, mat2)


def aten_broadcast_tensors(tensors: Sequence[TensorType]) -> TensorType:
    """broadcast_tensors(Tensor[] tensors) -> Tensor[]"""

    raise NotImplementedError()


@torch_op("aten::broadcast_to")
def aten_broadcast_to(self: TTensor, size: INT64) -> TTensor:
    """broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)"""

    return op.Expand(self, size)


def aten_bucketize(
    self: TensorType, boundaries: TensorType, out_int32: bool = False, right: bool = False
) -> TensorType:
    """bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor"""

    raise NotImplementedError()


def aten_can_cast(from_: int, to: int) -> bool:
    """can_cast(ScalarType from, ScalarType to) -> bool"""

    raise NotImplementedError()


def aten_cartesian_prod(tensors: Sequence[TensorType]) -> TensorType:
    """cartesian_prod(Tensor[] tensors) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::cat")
def aten_cat(tensors: Sequence[TTensor], dim: int = 0) -> TTensor:
    """cat(Tensor[] tensors, int dim=0) -> Tensor"""

    # NOTE: Having empty tensors when concatenating along non-zero dimension
    # is not supported.
    # TODO(justinchuby): Filter these tensors out with Sequence ops before
    # calling ConcatFromSequence.
    return op.ConcatFromSequence(tensors, axis=dim)


def aten_ccol_indices(self: TensorType) -> TensorType:
    """ccol_indices(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


def aten_ccol_indices_copy(self: TensorType) -> TensorType:
    """ccol_indices_copy(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_cdist(
    x1: TensorType, x2: TensorType, p: float = 2.0, compute_mode: Optional[int] = None
) -> TensorType:
    """cdist(Tensor x1, Tensor x2, float p=2, int? compute_mode=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::ceil")
def aten_ceil(self: TFloat) -> TFloat:
    """ceil(Tensor self) -> Tensor"""

    return op.Ceil(self)


def aten_chain_matmul(matrices: Sequence[TensorType]) -> TensorType:
    """chain_matmul(Tensor[] matrices) -> Tensor"""

    raise NotImplementedError()


def aten_chalf(self: TensorType, memory_format: Optional[str] = None) -> TensorType:
    """chalf(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""

    raise NotImplementedError()


def aten_channel_shuffle(self: TensorType, groups: int) -> TensorType:
    """channel_shuffle(Tensor self, int groups) -> Tensor"""

    raise NotImplementedError()


def aten_cholesky(self: TensorType, upper: bool = False) -> TensorType:
    """cholesky(Tensor self, bool upper=False) -> Tensor"""

    raise NotImplementedError()


def aten_cholesky_inverse(self: TensorType, upper: bool = False) -> TensorType:
    """cholesky_inverse(Tensor self, bool upper=False) -> Tensor"""

    raise NotImplementedError()


def aten_cholesky_solve(
    self: TensorType, input2: TensorType, upper: bool = False
) -> TensorType:
    """cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor"""

    raise NotImplementedError()


def aten_choose_qparams_optimized(
    input: TensorType, numel: int, n_bins: int, ratio: float, bit_width: int
) -> tuple[TensorType, TensorType]:
    """choose_qparams_optimized(Tensor input, int numel, int n_bins, float ratio, int bit_width) -> (Tensor, Tensor)"""

    raise NotImplementedError()


@torch_op("aten::chunk")
def aten_chunk(self: TTensor, chunks: int, dim: int = 0) -> Sequence[TTensor]:
    """chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]"""
    # This will create a Sequence of tensors
    neg_1 = op.Constant(value_ints=[-1])
    # Get size of specified dim
    self_shape = op.Shape(self)
    dim_size = op.Gather(self_shape, dim, axis=0)
    # Compute size/chunk to get the number of data in one chunk
    num_per_chunk = op.Div(dim_size, chunks)
    num_per_chunk = op.Cast(op.Mod(dim_size, chunks) > 0, to=INT64.dtype) + num_per_chunk  # type: ignore[operator]

    # Compute real chunk number
    num_chunk = op.Div(dim_size, num_per_chunk)
    # Get something like [n, n, n, n, ...], total num_chunk
    list_split = op.Expand(num_per_chunk, op.Reshape(num_chunk, neg_1))

    remainder = op.Mod(dim_size, num_per_chunk)
    if remainder > 0:  # type: ignore[operator]
        # Append the remainder to the [n, n, n, n, ..., r]
        list_split = op.Concat(list_split, op.Reshape(remainder, neg_1), axis=0)

    return op.SplitToSequence(self, list_split, axis=dim)


@torch_op(("aten::clamp", "aten::clamp.Tensor"), trace_only=True)
def aten_clamp(self: TReal, min: Optional[TReal] = None, max: Optional[TReal] = None) -> TReal:
    """clamp(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor"""
    clamped = self

    if min is None and max is None:
        return clamped

    # If min is greater than max torch.clamp(..., min, max)
    # sets all elements in input to the value of max.
    # So this order is important.
    if min is not None:
        min_clamp = op.CastLike(min, self)
        clamped = op.Max(clamped, min_clamp)

    if max is not None:
        max_clamp = op.CastLike(max, self)
        clamped = op.Min(clamped, max_clamp)

    return clamped


@torch_op("aten::clamp_max")
def aten_clamp_max(self: TReal, max_: TReal) -> TReal:
    """clamp_max(Tensor self, Tensor max) -> Tensor"""

    self_size = op.Size(self)
    max_shape = op.Shape(max_)
    max_rank = op.Size(max_shape)
    if self_size == 0:
        result = op.Expand(self, max_shape)
    else:
        if max_rank == 0:
            max_ = op.CastLike(max_, self)
            result = op.Clip(self, None, max_)
        else:
            result = op.Min(self, max_)

    return result


@torch_op("aten::clamp_min")
def aten_clamp_min(self: TReal, min_: TReal) -> TReal:
    """clamp_min(Tensor self, Tensor min) -> Tensor"""

    self_size = op.Size(self)
    min_shape = op.Shape(min_)
    min_rank = op.Size(min_shape)
    if self_size == 0:
        result = op.Expand(self, min_shape)
    else:
        if min_rank == 0:
            min_ = op.CastLike(min_, self)
            result = op.Clip(self, min_, None)
        else:
            result = op.Max(self, min_)

    return result


@torch_op("aten::clone")
def aten_clone(
    self: TTensor, memory_format: str = ""  # pylint: disable=unused-argument
) -> TTensor:
    """clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""

    return op.Identity(self)


def aten_coalesce(self: TensorType) -> TensorType:
    """coalesce(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


def aten_col_indices(self: TensorType) -> TensorType:
    """col_indices(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


def aten_col_indices_copy(self: TensorType) -> TensorType:
    """col_indices_copy(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_column_stack(tensors: Sequence[TensorType]) -> TensorType:
    """column_stack(Tensor[] tensors) -> Tensor"""

    raise NotImplementedError()


def aten_combinations(
    self: TensorType, r: int = 2, with_replacement: bool = False
) -> TensorType:
    """combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor"""

    raise NotImplementedError()


def aten_complex(real: TensorType, imag: TensorType) -> TensorType:
    """complex(Tensor real, Tensor imag) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::concat")
def aten_concat(tensors: Sequence[TTensor], dim: int = 0) -> TTensor:
    """concat(Tensor[] tensors, int dim=0) -> Tensor"""

    # TODO(justinchuby): Combine the implementation with cat
    return op.ConcatFromSequence(tensors, axis=dim)


@torch_op("aten::concatenate")
def aten_concatenate(tensors: Sequence[TTensor], dim: int = 0) -> TTensor:
    """concatenate(Tensor[] tensors, int dim=0) -> Tensor"""

    # TODO(justinchuby): Combine the implementation with cat
    return op.ConcatFromSequence(tensors, axis=dim)


@torch_op("aten::conj")
def aten_conj(self: TTensor) -> TTensor:
    """conj(Tensor(a) self) -> Tensor(a)"""

    return op.Identity(self)


@torch_op("aten::conj", complex=True, private=True)
def _complex_conjugate(self: TFloat) -> TFloat:
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    two = op.Constant(value_ints=[2])
    neg_1 = op.Constant(value_ints=[-1])
    # The last dimension is the real and imaginary parts

    real = op.Slice(self, zero, one, neg_1)
    imag = op.Slice(self, one, two, neg_1)
    conjugated = op.Concat(real, op.Neg(imag), axis=-1)

    return conjugated


@torch_op("aten::conj", complex=True, trace_only=True)
def aten_conj_complex(self: TFloat) -> TFloat:
    """conj(Tensor(a) self) -> Tensor(a)"""

    # TODO(#834): Allow calling scripted functions from other
    # scripted functions and remove trace only.
    return _complex_conjugate(self)


def aten_conj_physical(self: TensorType) -> TensorType:
    """conj_physical(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::constant_pad_nd")
def aten_constant_pad_nd(self: TTensor, pad: INT64, value: float = 0.0) -> TTensor:
    """constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor"""

    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    # assume zero-dimensions in the beginning
    # rank = len(self.shape)  # rank must be scalar
    # paddings = list(pad[:]) + [0] * (rank * 2 - len(pad))
    # reverse order and collate first beginnings and then ends
    # paddings = paddings[-2::-2] + paddings[-1::-2]

    neg_1 = op.Constant(value_ints=[-1])

    rank = op.Size(op.Shape(self))
    zero_count = op.Sub(op.Mul(rank, 2), op.Size(pad))
    zero_count = op.Reshape(zero_count, neg_1)
    zero = op.Constant(value_ints=[0])
    zeros = op.Expand(zero, zero_count)
    torch_paddings = op.Concat(pad, zeros, axis=0)
    size_d = op.Size(torch_paddings)
    steps = op.Constant(value_ints=[-2])

    starts = steps
    ends = op.Sub(starts, size_d)
    odd_elements = op.Slice(torch_paddings, starts, ends, zero, steps)

    starts = neg_1
    ends = op.Sub(starts, size_d)
    even_elements = op.Slice(torch_paddings, starts, ends, zero, steps)

    onnx_padding = op.Concat(odd_elements, even_elements, axis=0)
    return op.Pad(self, onnx_padding, value)


@torch_op("aten::contiguous")
def aten_contiguous(
    self: TTensor, memory_format: str = "contiguous_format"  # pylint: disable=unused-argument
) -> TTensor:
    """contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)"""

    # ONNX does not have the notion of memory_format. It is always treated as a no-op.
    return op.Identity(self)


@torch_op("aten::conv1d", trace_only=True)
def aten_conv1d(
    input: TFloat,
    weight: TFloat,
    bias: Optional[TFloat] = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    groups: int = 1,
) -> TFloat:
    """conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor"""

    # Attributes need to be manipulated in Python to match ONNX's conv1d
    if not isinstance(padding, Sequence):
        padding = (padding,)
    pads = [*padding, *padding]

    if not isinstance(dilation, Sequence):
        dilation = (dilation,)
    dilations = list(dilation)

    if not isinstance(stride, Sequence):
        stride = (stride,)
    strides = list(stride)

    if bias is None:
        weight_dim_0 = op.Shape(weight, start=0, end=1)
        bias_shape = op.Expand(weight_dim_0, op.Constant(value_ints=[1]))
        zero = op.CastLike(0.0, input)
        bias = op.Expand(zero, bias_shape)

    result = _aten_convolution_onnx(
        input,
        weight,
        bias,
        transposed=False,
        strides=strides,
        pads=pads,
        dilations=dilations,
        groups=groups,
    )

    return result


@torch_op("aten::conv2d", trace_only=True)
def aten_conv2d(
    input: TFloat,
    weight: TFloat,
    bias: Optional[TFloat] = None,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
) -> TFloat:
    """conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor"""

    # Attributes need to be manipulated in Python to match ONNX's conv2d
    if not isinstance(padding, Sequence):
        padding = (padding, padding)
    pads = [*padding, *padding]

    if not isinstance(dilation, Sequence):
        dilation = (dilation, dilation)
    dilations = list(dilation)

    if not isinstance(stride, Sequence):
        stride = (stride, stride)
    strides = list(stride)

    if bias is None:
        weight_dim_0 = op.Shape(weight, start=0, end=1)
        bias_shape = op.Expand(weight_dim_0, op.Constant(value_ints=[1]))
        zero = op.CastLike(0.0, input)
        bias = op.Expand(zero, bias_shape)

    result = _aten_convolution_onnx(
        input,
        weight,
        bias,
        transposed=False,
        strides=strides,
        pads=pads,
        dilations=dilations,
        groups=groups,
    )

    return result


@torch_op("aten::conv3d", trace_only=True)
def aten_conv3d(
    input: TFloat,
    weight: TFloat,
    bias: Optional[TFloat] = None,
    stride: Sequence[int] = (1, 1, 1),
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    groups: int = 1,
) -> TFloat:
    """conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor"""

    # Attributes need to be manipulated in Python to match ONNX's conv3d
    if not isinstance(padding, Sequence):
        padding = (padding, padding, padding)
    pads = [*padding, *padding]

    if not isinstance(dilation, Sequence):
        dilation = (dilation, dilation, dilation)
    dilations = list(dilation)

    if not isinstance(stride, Sequence):
        stride = (stride, stride, stride)
    strides = list(stride)

    if bias is None:
        weight_dim_0 = op.Shape(weight, start=0, end=1)
        bias_shape = op.Expand(weight_dim_0, op.Constant(value_ints=[1]))
        zero = op.CastLike(0.0, input)
        bias = op.Expand(zero, bias_shape)

    result = _aten_convolution_onnx(
        input,
        weight,
        bias,
        transposed=False,
        strides=strides,
        pads=pads,
        dilations=dilations,
        groups=groups,
    )

    return result


def aten_conv_tbc(
    self: TensorType, weight: TensorType, bias: TensorType, pad: int = 0
) -> TensorType:
    """conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor"""

    raise NotImplementedError()


def aten_conv_tbc_backward(
    self: TensorType, input: TensorType, weight: TensorType, bias: TensorType, pad: int
) -> tuple[TensorType, TensorType, TensorType]:
    """conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)"""

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
    """conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::convolution", trace_only=True)
def aten_convolution(
    input: TFloat,
    weight: TFloat,
    bias: Optional[TFloat] = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    transposed: bool = False,
    output_padding: Sequence[int] = (0,),
    groups: int = 1,
) -> TFloat:
    """convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups) -> Tensor"""

    if not isinstance(padding, Sequence):
        padding = (padding, padding)
    pads = [*padding, *padding]

    if not isinstance(dilation, Sequence):
        dilation = (dilation, dilation)
    dilations = list(dilation)

    if not isinstance(stride, Sequence):
        stride = (stride, stride)
    strides = list(stride)

    if bias is None:
        weight_dim_0 = op.Shape(weight, start=0, end=1)
        bias_shape = op.Expand(weight_dim_0, op.Constant(value_ints=[1]))
        zero = op.CastLike(0.0, input)
        bias = op.Expand(zero, bias_shape)

    result = _aten_convolution_onnx(
        input,
        weight,
        bias,
        transposed,
        strides=strides,
        pads=pads,
        dilations=dilations,
        output_padding=output_padding,
        groups=groups,
    )

    return result


@torch_op("aten::convolution", private=True)
def _aten_convolution_onnx(
    input: TFloat,
    weight: TFloat,
    bias: TFloat,
    transposed: BOOL,
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    output_padding: Sequence[int] = (0,),
    groups: int = 1,
) -> TFloat:
    """ConvXd with attributes pre-computed to fit the ONNX spec."""

    # NOTE: transposed must be an input because when provided as an attribute,
    # it will be an integer, not a boolean, which will fail the if condition.
    # Alternatively we could cast transposed to BOOL.
    # E.g. `if op.Cast(transposed, BOOL.dtype): ...`

    weight_size = op.Size(op.Shape(weight))
    no_batch = op.Size(op.Shape(input)) != weight_size

    if no_batch:
        input = op.Unsqueeze(input, op.Constant(value_ints=[0]))

    if transposed:
        result = op.ConvTranspose(
            input,
            weight,
            bias,
            strides=strides,
            pads=pads,
            group=groups,
            dilations=dilations,
            output_padding=output_padding,
        )
    else:
        result = op.Conv(
            input,
            weight,
            bias,
            strides=strides,
            pads=pads,
            group=groups,
            dilations=dilations,
        )

    if no_batch:
        result = op.Squeeze(result, op.Constant(value_ints=[0]))

    return result


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
    """convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""

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
    """convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)"""

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
    """convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::copy")
def aten_copy(
    self: TTensor, src: TTensor, non_blocking: bool = False  # pylint: disable=unused-argument
) -> TTensor:
    """copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor"""

    self = op.Identity(src)
    return self


def aten_copysign(self: TensorType, other: TensorType) -> TensorType:
    """copysign.Tensor(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_corrcoef(self: TensorType) -> TensorType:
    """corrcoef(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::cos")
def aten_cos(self: TFloat) -> TFloat:
    """cos(Tensor self) -> Tensor"""

    return op.Cos(self)


@torch_op("aten::cosh")
def aten_cosh(self: TFloat) -> TFloat:
    """cosh(Tensor self) -> Tensor"""

    return op.Cosh(self)


def aten_cosine_embedding_loss(
    input1: TensorType,
    input2: TensorType,
    target: TensorType,
    margin: float = 0.0,
    reduction: int = 1,
) -> TensorType:
    """cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


def aten_cosine_similarity(
    x1: TensorType, x2: TensorType, dim: int = 1, eps: float = 1e-08
) -> TensorType:
    """cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor"""

    raise NotImplementedError()


def aten_count_nonzero(self: TensorType, dim: Optional[int] = None) -> TensorType:
    """count_nonzero(Tensor self, int? dim=None) -> Tensor"""

    raise NotImplementedError()


def aten_cov(
    self: TensorType,
    correction: int = 1,
    fweights: Optional[TensorType] = None,
    aweights: Optional[TensorType] = None,
) -> TensorType:
    """cov(Tensor self, *, int correction=1, Tensor? fweights=None, Tensor? aweights=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::cross")
def aten_cross(self: TTensor, other: TTensor, dim: int = -1) -> TTensor:
    """cross(Tensor self, Tensor other, int? dim=None) -> Tensor"""

    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    two = op.Constant(value_ints=[2])
    three = op.Constant(value_ints=[3])
    axes = op.Expand(dim, op.Constant(value_ints=[1]))

    # Reference https://en.wikipedia.org/w/index.php?title=Cross_product&oldid=1143125073
    a1 = op.Slice(self, zero, one, axes)
    a2 = op.Slice(self, one, two, axes)
    a3 = op.Slice(self, two, three, axes)
    b1 = op.Slice(other, zero, one, axes)
    b2 = op.Slice(other, one, two, axes)
    b3 = op.Slice(other, two, three, axes)
    # Broadcasting is implicitly supported by Mul
    c1 = op.Sub(op.Mul(a2, b3), op.Mul(a3, b2))
    c2 = op.Sub(op.Mul(a3, b1), op.Mul(a1, b3))
    c3 = op.Sub(op.Mul(a1, b2), op.Mul(a2, b1))

    return op.Concat(c1, c2, c3, axis=dim)


def aten_crow_indices(self: TensorType) -> TensorType:
    """crow_indices(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


def aten_crow_indices_copy(self: TensorType) -> TensorType:
    """crow_indices_copy(Tensor self) -> Tensor"""

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
    """ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor"""

    raise NotImplementedError()


def aten_cudnn_affine_grid_generator(
    theta: TensorType, N: int, C: int, H: int, W: int
) -> TensorType:
    """cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid"""

    raise NotImplementedError()


def aten_cudnn_affine_grid_generator_backward(
    grad: TensorType, N: int, C: int, H: int, W: int
) -> TensorType:
    """cudnn_affine_grid_generator_backward(Tensor grad, int N, int C, int H, int W) -> Tensor grad_theta"""

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
    """cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)"""

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
    """cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace) -> (Tensor, Tensor, Tensor)"""

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
    """cudnn_convolution(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor"""

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
    """cudnn_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor"""

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
    """cudnn_convolution_relu(Tensor self, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor"""

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
    """cudnn_convolution_transpose(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor"""

    raise NotImplementedError()


def aten_cudnn_grid_sampler(self: TensorType, grid: TensorType) -> TensorType:
    """cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output"""

    raise NotImplementedError()


def aten_cudnn_grid_sampler_backward(
    self: TensorType, grid: TensorType, grad_output: TensorType
) -> tuple[TensorType, TensorType]:
    """cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor grad_self, Tensor grad_grid)"""

    raise NotImplementedError()


def aten_cudnn_is_acceptable(self: TensorType) -> bool:
    """cudnn_is_acceptable(Tensor self) -> bool"""

    raise NotImplementedError()


def aten_cummax(self: TensorType, dim: int) -> tuple[TensorType, TensorType]:
    """cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)"""

    raise NotImplementedError()


def aten_cummaxmin_backward(
    grad: TensorType, input: TensorType, indices: TensorType, dim: int
) -> TensorType:
    """cummaxmin_backward(Tensor grad, Tensor input, Tensor indices, int dim) -> Tensor"""

    raise NotImplementedError()


def aten_cummin(self: TensorType, dim: int) -> tuple[TensorType, TensorType]:
    """cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)"""

    raise NotImplementedError()


def aten_cumprod(self: TensorType, dim: int, dtype: Optional[int] = None) -> TensorType:
    """cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""

    raise NotImplementedError()


def aten_cumprod_backward(
    grad: TensorType, input: TensorType, dim: int, output: TensorType
) -> TensorType:
    """cumprod_backward(Tensor grad, Tensor input, int dim, Tensor output) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::cumsum", trace_only=True)
def aten_cumsum(
    self: TRealUnlessInt16OrInt8, dim: Union[INT32, INT64], dtype: int = -1
) -> TRealUnlessInt16OrInt8:
    """cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""

    # TODO(justinchuby): The accumulation type for int32 is int64. Consider excluding from inputs.
    if dtype == -1:
        cast = self
    else:
        cast = op.Cast(self, to=dtype)
    return _aten_cumsum_onnx(cast, dim)


@torch_op("aten::cumsum", private=True)
def _aten_cumsum_onnx(
    self: TRealUnlessInt16OrInt8, dim: Union[INT32, INT64]
) -> TRealUnlessInt16OrInt8:
    if op.Size(op.Shape(self)) == 0:
        # A scalar
        result = op.Identity(self)
    else:
        result = op.CumSum(self, dim)
    return result


def aten_data(self: TensorType) -> TensorType:
    """data(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_deg2rad(self: TensorType) -> TensorType:
    """deg2rad(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_dense_dim(self: TensorType) -> int:
    """dense_dim(Tensor self) -> int"""

    raise NotImplementedError()


def aten_det(self: TensorType) -> TensorType:
    """det(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::detach")
def aten_detach(self: TensorType) -> TensorType:
    """detach(Tensor(a) self) -> Tensor(a)"""

    return op.Identity(self)


def aten_detach_copy(self: TensorType) -> TensorType:
    """detach_copy(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_diag(self: TensorType, diagonal: int = 0) -> TensorType:
    """diag(Tensor self, int diagonal=0) -> Tensor"""

    raise NotImplementedError()


def aten_diag_embed(
    self: TensorType, offset: int = 0, dim1: int = -2, dim2: int = -1
) -> TensorType:
    """diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor"""

    raise NotImplementedError()


def aten_diagflat(self: TensorType, offset: int = 0) -> TensorType:
    """diagflat(Tensor self, int offset=0) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::diagonal", trace_only=True)
def aten_diagonal(self: TReal, offset: int = 0, dim1: int = 0, dim2: int = 1) -> TReal:
    """diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)"""

    # perm is used to transpose the tensor to make dim1 and dim2 as the last 2 dims
    # [0,1,2] -> [2,0,1] when dim1=0 and dim2=1
    # [0,1,2] -> [1,0,2] when dim1=0 and dim2=2
    # [0,1,2] -> [0,1,2] when dim1=1 and dim2=2
    self_rank = len(self.shape)
    perm = list(range(self_rank))
    perm.remove(dim1)
    perm.remove(dim2)
    perm.append(dim1)
    perm.append(dim2)

    # If rank=2, then axes=[0]; if rank=3, then axes=[1]
    # This is because computing diagonal sum is on dim2 after transpose by perm
    axes = [self_rank - 2]

    return _aten_diagonal_onnx(self, offset, dim1, dim2, perm, axes)


@torch_op("aten::diagonal", private=True)
def _aten_diagonal_onnx(
    self: TTensor, offset: int, dim1: int, dim2: int, perm: Sequence[int], axes: Sequence[int]
) -> TTensor:
    neg_1 = op.Constant(value_ints=[-1])
    dim1_size = op.Reshape(op.Gather(op.Shape(self), dim1), neg_1)  # row
    dim2_size = op.Reshape(op.Gather(op.Shape(self), dim2), neg_1)  # col
    mask_shape = op.Concat(dim1_size, dim2_size, axis=0)
    tmp_tensor = op.ConstantOfShape(mask_shape)
    mask = op.EyeLike(tmp_tensor, k=offset)
    mask = op.CastLike(mask, self)
    self_t = op.Transpose(self, perm=perm)
    result = op.Mul(self_t, mask)
    result = op.ReduceSum(result, keepdims=False, axes=axes)
    # min(row, col)
    min_dim_size = op.Min(dim1_size, dim2_size)
    # take 2 tensors as example:
    # one is 3x5 in size, min_dim_size = 3, dim1_size = 3
    # the other is 5x3 in size, min_dim_size = 3, dim1_size = 5
    # 3 rows x 5 cols     5 rows x 3 cols
    # offset  diagonal    offset  diagonal
    # ----------------    ----------------
    # -4      0           -6      0
    # -3      0           -5      0
    # -2      1           -4      1
    # -1      2           -3      2
    # 0       3           -2      3
    # 1       3           -1      3
    # 2       3           0       3
    # 3       2           1       2
    # 4       1           2       1
    # 5       0           3       0
    # 6       0           4       0

    # From above table, we can get the logic below
    if offset < 0:
        # row + offset
        length = dim1_size + offset
        start = op.Constant(value_ints=[0])
    else:  # offset >= 0
        # col - offset
        length = dim2_size - offset
        start = op.Reshape(op.Constant(value_int=offset), neg_1)

    # max(min(length, min(row, col)), 0)
    length = op.Max(op.Min(length, min_dim_size), 0)
    end = start + length
    result = op.Slice(result, start, end, axes=axes)

    return result


@torch_op("aten::diagonal", trace_only=True)
def aten_diagonal_bool(self: BOOL, offset: int = 0, dim1: int = 0, dim2: int = 1) -> BOOL:
    """diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)"""

    # perm is used to transpose the tensor to make dim1 and dim2 as the last 2 dims
    # [0,1,2] -> [2,0,1] when dim1=0 and dim2=1
    # [0,1,2] -> [1,0,2] when dim1=0 and dim2=2
    # [0,1,2] -> [0,1,2] when dim1=1 and dim2=2
    self_rank = len(self.shape)
    perm = list(range(self_rank))
    perm.remove(dim1)
    perm.remove(dim2)
    perm.append(dim1)
    perm.append(dim2)

    # If rank=2, then axes=[0]; if rank=3, then axes=[1]
    # This is because computing diagonal sum is on dim2 after transpose by perm
    axes = [self_rank - 2]

    return _aten_diagonal_bool_onnx(self, offset, dim1, dim2, perm, axes)


@torch_op("aten::diagonal", private=True)
def _aten_diagonal_bool_onnx(
    self: BOOL, offset: int, dim1: int, dim2: int, perm: Sequence[int], axes: Sequence[int]
) -> BOOL:
    neg_1 = op.Constant(value_ints=[-1])
    dim1_size = op.Reshape(op.Gather(op.Shape(self), dim1), neg_1)  # row
    dim2_size = op.Reshape(op.Gather(op.Shape(self), dim2), neg_1)  # col
    mask_shape = op.Concat(dim1_size, dim2_size, axis=0)
    tmp_tensor = op.ConstantOfShape(mask_shape)
    mask = op.EyeLike(tmp_tensor, k=offset)
    self_int = op.Cast(self, to=INT64.dtype)
    mask_int = op.Cast(mask, to=INT64.dtype)
    self_int_t = op.Transpose(self_int, perm=perm)
    result = op.Mul(self_int_t, mask_int)
    result = op.ReduceSum(result, keepdims=False, axes=axes)
    # min(row, col)
    min_dim_size = op.Min(dim1_size, dim2_size)
    # take 2 tensors as example:
    # one is 3x5 in size, min_dim_size = 3, dim1_size = 3
    # the other is 5x3 in size, min_dim_size = 3, dim1_size = 5
    # 3 rows x 5 cols     5 rows x 3 cols
    # offset  diagonal    offset  diagonal
    # ----------------    ----------------
    # -4      0           -6      0
    # -3      0           -5      0
    # -2      1           -4      1
    # -1      2           -3      2
    # 0       3           -2      3
    # 1       3           -1      3
    # 2       3           0       3
    # 3       2           1       2
    # 4       1           2       1
    # 5       0           3       0
    # 6       0           4       0

    # From above table, we can get the logic below
    if offset < 0:
        # row + offset
        length = dim1_size + offset
        start = op.Constant(value_ints=[0])
    else:  # offset >= 0
        # col - offset
        length = dim2_size - offset
        start = op.Reshape(op.Constant(value_int=offset), neg_1)

    # max(min(length, min(row, col)), 0)
    length = op.Max(op.Min(length, min_dim_size), 0)
    end = start + length
    result = op.Slice(result, start, end, axes=axes)
    result = op.Cast(result, to=BOOL.dtype)

    return result


def aten_diagonal_backward(
    grad_output: TensorType, input_sizes: INT64, offset: int, dim1: int, dim2: int
) -> TensorType:
    """diagonal_backward(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2) -> Tensor"""

    raise NotImplementedError()


def aten_diagonal_copy(
    self: TensorType, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> TensorType:
    """diagonal_copy(Tensor self, int offset=0, int dim1=0, int dim2=1) -> Tensor"""

    raise NotImplementedError()


def aten_diagonal_scatter(
    self: TensorType, src: TensorType, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> TensorType:
    """diagonal_scatter(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1) -> Tensor"""

    raise NotImplementedError()


def aten_diff(
    self: TensorType,
    n: int = 1,
    dim: int = -1,
    prepend: Optional[TensorType] = None,
    append: Optional[TensorType] = None,
) -> TensorType:
    """diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor"""

    raise NotImplementedError()


def aten_digamma(self: TensorType) -> TensorType:
    """digamma(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_dist(self: TensorType, other: TensorType, p: float = 2.0) -> TensorType:
    """dist(Tensor self, Tensor other, Scalar p=2) -> Tensor"""

    raise NotImplementedError()


@torch_op(
    (
        "aten::div",
        "aten::div.Tensor",
        "aten::div.Scalar",
        # When rounding_mode is None, performs a true division
        # https://pytorch.org/docs/stable/generated/torch.div.html
        "aten::div.Tensor_mode",
        "aten::div.Scalar_mode",
        "aten::divide",
        "aten::true_divide",
    )
)
def aten_div(self: TFloat, other: TFloat) -> TFloat:
    """div.Tensor(Tensor self, Tensor other) -> Tensor"""

    # Int inputs will be promoted to float by PyTorch
    return op.Div(self, other)


@torch_op(("aten::div.Tensor_mode", "aten::div.Scalar_mode"), trace_only=True)
def aten_div_mode(self: TFloat, other: TFloat, rounding_mode: str) -> TFloat:
    """div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor"""

    # TODO(justinchuby): trace_only=False when we use opset19 which supports string comparison
    assert rounding_mode in {"trunc", "floor"}

    if rounding_mode == "trunc":
        # Rounds the results of the division towards zero.
        # Equivalent to C-style integer division
        result = aten_trunc(op.Div(self, other))
    else:  # rounding_mode == "floor"
        result = op.Floor(op.Div(self, other))

    return result


@torch_op(("aten::div.Tensor_mode", "aten::div.Scalar_mode"), trace_only=True)
def aten_div_mode_int(self: TInt, other: TInt, rounding_mode: str) -> TInt:
    """div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor

    Variant for integer inputs.
    """
    # TODO(justinchuby): trace_only=False when we use opset19 which supports string comparison
    assert rounding_mode in {"trunc", "floor"}

    quotient = op.Div(op.Cast(self, to=FLOAT.dtype), op.Cast(other, to=FLOAT.dtype))

    if rounding_mode == "trunc":
        # Rounds the results of the division towards zero.
        # Equivalent to C-style integer division
        result = aten_trunc(quotient)
    else:  # rounding_mode == "floor"
        result = op.Floor(quotient)

    return op.CastLike(result, self)


@torch_op("aten::dot")
def aten_dot(self: TFloat, tensor: TFloat) -> TFloat:
    """dot(Tensor self, Tensor tensor) -> Tensor"""

    return op.MatMul(self, tensor)


@torch_op("aten::dropout")
def aten_dropout(input: TFloat, p: FLOAT, train: BOOL) -> TFloat:
    """dropout(Tensor input, float p, bool train) -> Tensor"""

    input_is_scalar = op.Size(op.Shape(input)) == 0
    if input_is_scalar:
        input = op.Reshape(input, op.Constant(value_ints=[-1]))
        result, _ = op.Dropout(input, p, train)
        result = op.Squeeze(result)
    else:
        result, _ = op.Dropout(input, p, train)

    return result


def aten_dstack(tensors: Sequence[TensorType]) -> TensorType:
    """dstack(Tensor[] tensors) -> Tensor"""

    raise NotImplementedError()


def aten_einsum(
    equation: str, tensors: Sequence[TensorType], path: Optional[int] = None
) -> TensorType:
    """einsum(str equation, Tensor[] tensors, *, int[]? path=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::embedding")
def aten_embedding(
    weight: TTensor,
    indices: TTensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
):  # pylint: disable=unused-argument
    # embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor

    return op.Gather(weight, indices)


def aten_embedding_backward(
    grad: TensorType,
    indices: TensorType,
    num_weights: INT64,
    padding_idx: int,
    scale_grad_by_freq: bool,
    sparse: bool,
) -> TensorType:
    """embedding_backward(Tensor grad, Tensor indices, SymInt num_weights, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::embedding_bag", trace_only=True)
def aten_embedding_bag(
    weight: TFloat,
    indices: INT64,
    offsets: INT64,
    scale_grad_by_freq: bool = False,  # pylint: disable=unused-argument
    mode: int = 0,  # [0,1,2] indicate ["sum", "mean", "max"]
    sparse: bool = False,  # pylint: disable=unused-argument
    per_sample_weights: Optional[TFloat] = None,
    include_last_offset: bool = False,
) -> Tuple[TFloat, TFloat, TFloat, TFloat]:
    """embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)"""

    # assert(rank(indices) in [1,2])
    # assert(rank(offsets) == 1)
    # assert(op.Size(per_sample_weights) == op.Size(indices))
    if per_sample_weights is None:
        # Set per_sample_weights to 1.0, because cannot check 'None' in ONNX-Script
        # Size of persample_weights is the same as indices, and should be 1d tensor
        indices_1d = op.Reshape(indices, [-1])
        per_sample_weights = op.Expand(1, op.Shape(indices_1d))
        # Dtype of per_sample_weights is the same as weight
        per_sample_weights = op.CastLike(per_sample_weights, weight)

    result, offset2bag, bag_size, max_indices = _aten_embedding_bag_onnx(
        weight, indices, offsets, mode, per_sample_weights, include_last_offset
    )
    return result, offset2bag, bag_size, max_indices


@torch_op("aten::embedding_bag", private=True)
def _aten_embedding_bag_onnx(
    weight: TFloat,
    indices: INT64,
    offsets: INT64,
    mode: int,
    per_sample_weights: TFloat,
    include_last_offset: bool,
) -> Tuple[TFloat, TFloat, TFloat, TFloat]:
    neg_1 = op.Constant(value_ints=[-1])
    # Assume indices is shape(5,2), indices_1d is shape(10,)
    indices_1d = op.Reshape(indices, neg_1)
    # Get weight out according to indices_1d,
    new_weight = op.Gather(weight, indices_1d)
    # This happends after first step of Gather. Because Shape(indices)==Shape(per_sample_weights)
    new_weight = op.Mul(new_weight, op.Unsqueeze(per_sample_weights, axes=1))
    weight_dim_1 = op.Reshape(op.Shape(weight, start=1), neg_1)
    indices_size = op.Shape(indices_1d)

    # Assume indices is shape(5,2) reshape to (10,), offsets=[0,2,3], include_last_offset = False
    # [0,2,3] -> [0:2], [2:3], [3:10]
    num_bag = op.Reshape(op.Size(offsets), neg_1)  # 3 bags, means 10 is the last index
    if op.Equal(include_last_offset, True):
        num_bag = num_bag - 1  # 2 bags, means 3 is the last index
    else:
        offsets = op.Concat(offsets, indices_size, axis=0)  # Replace end with number

    # The element in sequence must be FLOAT32 dtype due to ORT bug
    new_weight = op.Cast(new_weight, to=FLOAT.dtype)
    # FIXME: https://github.com/microsoft/onnxruntime/issues/16846
    result = op.SequenceEmpty()

    index_tensor = op.Reshape(op.Constant(value_int=0), neg_1)  # Used for iterator
    cond = index_tensor < num_bag
    # Process each bag
    while cond:
        start = op.Slice(offsets, index_tensor, index_tensor + 1)
        end = op.Slice(offsets, index_tensor + 1, index_tensor + 2)
        # row_result should be 0, need to generate (1,N) shape tensor with 0 values
        if start == end:
            row_result = op.Expand(
                op.Constant(value_floats=[0.0]),
                op.Concat(op.Constant(value_ints=[1]), weight_dim_1, axis=0),
            )
        else:
            if mode == 0:  # sum
                weight_rows = op.Slice(new_weight, start, end)
                row_result = op.ReduceSum(weight_rows, axes=[0])
            elif mode == 1:  # mean
                weight_rows = op.Slice(new_weight, start, end)
                if op.Equal(index_tensor, num_bag - 1):  # The last bag
                    row_result = op.ReduceSum(weight_rows, axes=[0])
                    # When include_last_offset=False, offsets=[0,2,3] -> [0,2,3,10], denominator=10-3=7
                    # When include_last_offset=True, offsets=[0,2,3], denominator=10-2=8
                    denominator = op.Sub(op.Shape(indices, start=0, end=1), start)
                    if op.Greater(denominator, 0):
                        row_result = op.Div(row_result, op.CastLike(denominator, new_weight))
                else:
                    row_result = op.ReduceMean(weight_rows, axes=[0])
            else:  # max
                if op.Equal(index_tensor, num_bag - 1):  # The last bag
                    weight_rows = op.Slice(new_weight, start, indices_size)
                else:
                    weight_rows = op.Slice(new_weight, start, end)
                row_result = op.ReduceMax(weight_rows, axes=[0])

        result = op.SequenceInsert(result, row_result)
        index_tensor = index_tensor + 1
        cond = index_tensor < num_bag

    result = op.ConcatFromSequence(result, axis=0)
    result = op.CastLike(result, weight)

    # Only compute the shape of other 3 outputs, we don't care the value
    if mode == 0:  # sum
        offset2bag = op.Shape(indices, start=0, end=0)  # Generate empty tensor
        if op.Equal(include_last_offset, True):
            bag_size = op.Expand(0, op.Shape(offsets))
        else:
            bag_size = op.Expand(0, op.Shape(offsets) - 1)
        max_indices = op.Expand(0, op.Shape(bag_size))
    elif mode == 1:  # mean
        offset2bag = op.Expand(0, op.Shape(indices, start=0, end=1))
        bag_size = op.Expand(0, op.Shape(offsets) - 1)
        max_indices = op.Expand(0, op.Shape(bag_size))
    else:  # max
        offset2bag = op.Expand(0, op.Shape(indices, start=0, end=1))
        bag_size = op.Expand(0, op.Shape(offsets) - 1)
        # shape = (bag_size.dim[0], weight.dim[1])
        dim_0 = op.Shape(bag_size, start=0, end=1)
        dim_1 = op.Shape(weight, start=1, end=2)
        max_indices = op.Expand(0, op.Concat(dim_0, dim_1, axis=0))

    return result, offset2bag, bag_size, max_indices


@torch_op(
    (
        "aten::embedding_bag.padding_idx",
        "aten::_embedding_bag",
        "aten::_embedding_bag_forward_only",
    ),
    trace_only=True,
)
def aten_embedding_bag_padding_idx(
    weight: TFloat,
    indices: INT64,
    offsets: INT64,
    scale_grad_by_freq: bool = False,  # pylint: disable=unused-argument
    mode: int = 0,  # [0,1,2] indicate ["sum", "mean", "max"]
    sparse: bool = False,  # pylint: disable=unused-argument
    per_sample_weights: Optional[TFloat] = None,
    include_last_offset: bool = False,
    padding_idx: int = -1,
) -> Tuple[TFloat, TFloat, TFloat, TFloat]:
    """embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)

    We add default values for the attributes to accommodate _embedding_bag as well:
    _embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1)
    """
    assert (
        padding_idx is not None
    ), "padding_idx must not be None. This is likely a dispatcher error"

    if per_sample_weights is None:
        per_sample_weights = op.Expand(op.Constant(value_floats=[1.0]), op.Shape(indices))
        per_sample_weights = op.CastLike(per_sample_weights, weight)

    # Change padding_idx to positive value, -1 means the last index
    if padding_idx < 0:
        padding_idx = weight.shape[0] + padding_idx

    result, offset2bag, bag_size, max_indices = _aten_embedding_bag_1d_padding_idx_onnx(
        weight, indices, offsets, mode, per_sample_weights, include_last_offset, padding_idx
    )

    return result, offset2bag, bag_size, max_indices


@torch_op("aten::embedding_bag.padding_idx", private=True)
def _aten_embedding_bag_1d_padding_idx_onnx(
    weight: TFloat,
    indices: INT64,
    offsets: INT64,
    mode: int,
    per_sample_weights: TFloat,
    include_last_offset: bool,
    padding_idx: int,
) -> Tuple[TFloat, TFloat, TFloat, TFloat]:
    neg_1 = op.Constant(value_ints=[-1])
    # Get weight out according to indices,
    # e.g. indices=[3,1,4,5,3] means get weight[[3,1,4,5,3]]
    indices_weight = op.Gather(weight, indices)
    # This happends after first step of Gather. Because Shape(indices)==Shape(per_sample_weights)
    indices_weight = op.Mul(indices_weight, op.Unsqueeze(per_sample_weights, axes=1))

    # The element in sequence must be FLOAT32 dtype due to ORT bug
    indices_weight = op.Cast(indices_weight, to=FLOAT.dtype)
    # FIXME: https://github.com/microsoft/onnxruntime/issues/16846
    result = op.SequenceEmpty()

    num_bag = op.Reshape(op.Size(offsets), neg_1)
    idx_size = op.Reshape(op.Size(indices), neg_1)

    if op.Equal(include_last_offset, True):
        num_bag = num_bag - 1
        # Change(by ScatterElement setting) the last element to 'end'
        # [0,2,3] -> [0,2,end]
        offsets = op.ScatterElements(offsets, [-1], idx_size)
    else:
        # Change [0,2,3] -> [0,2,3,end], means [0:2],[2:3],[3:end]
        offsets = op.Concat(offsets, idx_size, axis=0)

    # Process each bag
    i = op.Reshape(op.Constant(value_int=0), neg_1)  # Used for iterator
    cond_1 = i < num_bag
    while cond_1:
        start_pos = op.Gather(offsets, i)
        end_pos = op.Gather(offsets, i + 1)
        # empty tensor
        curr_offsets = op.Shape(indices, start=0, end=0)
        j = start_pos
        cond_2 = j < end_pos
        while cond_2:
            index = op.Gather(indices, j)
            if not op.Equal(index, padding_idx):
                # Something like the 'append' operation
                curr_offsets = op.Concat(curr_offsets, op.Reshape(j, neg_1), axis=0)
            j = j + 1
            cond_2 = j < end_pos

        # Empty input get zero value output, not empty output
        if op.Size(curr_offsets) == 0:
            dim_1 = op.Shape(weight, start=1, end=2)
            expand_shape = op.Concat([1], dim_1, axis=0)
            row_result = op.Expand([0.0], expand_shape)
        else:
            row_weight = op.Gather(indices_weight, curr_offsets)
            if mode == 0:  # sum
                row_result = op.ReduceSum(row_weight, axes=[0])
            elif mode == 1:  # mean
                row_result = op.ReduceMean(row_weight, axes=[0])
            else:
                row_result = op.ReduceMax(row_weight, axes=[0])

        result = op.SequenceInsert(result, row_result)

        i = i + 1
        cond_1 = i < num_bag

    result = op.ConcatFromSequence(result, axis=0)
    result = op.CastLike(result, weight)

    if mode == 0:  # sum
        offset2bag = op.Expand(0, op.Shape(indices))
        if op.Equal(include_last_offset, True):
            bag_size = op.Expand(0, op.Shape(offsets))
        else:
            bag_size = op.Expand(0, op.Shape(offsets) - 1)
        max_indices = op.Expand(0, op.Shape(bag_size))
    elif mode == 1:  # mean
        offset2bag = op.Expand(0, op.Shape(indices, start=0, end=1))
        bag_size = op.Expand(0, op.Shape(offsets) - 1)
        max_indices = op.Expand(0, op.Shape(bag_size))
    else:  # mode == 2, max
        offset2bag = op.Expand(0, op.Shape(indices, start=0, end=1))
        bag_size = op.Expand(0, op.Shape(offsets) - 1)
        # shape = (bag_size.dim[0], weight.dim[1])
        dim_0 = op.Shape(bag_size, start=0, end=1)
        dim_1 = op.Shape(weight, start=1, end=2)
        max_indices = op.Expand(0, op.Concat(dim_0, dim_1, axis=0))

    return result, offset2bag, bag_size, max_indices


def aten_embedding_dense_backward(
    grad_output: TensorType,
    indices: TensorType,
    num_weights: INT64,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> TensorType:
    """embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::embedding_renorm", trace_only=True)
def aten_embedding_renorm(
    weight: TFloat, indices: INT64, max_norm: float, norm_type: float = 2.0
) -> TFloat:
    """embedding_renorm(Tensor weight, Tensor indices, float max_norm, float norm_type) -> Tensor"""

    unique_indices = op.Unique(indices)
    unique_indices_Y = op.SequenceAt(unique_indices, 0)
    # using _onnx private function because op.SrquenceAt(unique_indices, 0) cannot pass module checker
    # The error message is:
    # onnx.onnx_cpp2py_export.shape_inference.InferenceError:
    # [ShapeInferenceError] Shape inference error(s): (op_type:aten_embedding_renorm,
    # node name: aten_embedding_renorm_0): [ShapeInferenceError] (op_type:SequenceAt,
    # node name: n2): input_sequence typestr: S, has unsupported type: tensor(int64)
    return aten_embedding_renorm_onnx(weight, unique_indices_Y, max_norm, norm_type)


@torch_op("aten::embedding_renorm", private=True)
def aten_embedding_renorm_onnx(
    weight: TFloat, indices: INT64, max_norm: float, norm_type: float = 2.0
) -> TFloat:
    partial_weight = op.Gather(weight, indices)
    # partial_weight_norm = sum(|w|^p)^(1/p)
    if norm_type == 1.0:
        # This is not necessary, but op.ReduceL1 is faster than function list in 'else'
        partial_weight_norm = op.ReduceL1(partial_weight, axes=[1], keepdims=True)
    elif norm_type == 2.0:
        # This is not necessary, but op.ReduceL2 is faster than function list in 'else'
        partial_weight_norm = op.ReduceL2(partial_weight, axes=[1], keepdims=True)
    else:
        # Abs -> Pow -> ReduceSum -> Pow -> Pow
        partial_weight_abs = op.Abs(partial_weight)
        partial_weight_pow = op.Pow(partial_weight_abs, op.Constant(value_float=norm_type))
        partial_weight_norm = op.ReduceSum(partial_weight_pow, axes=[1], keepdims=True)
        pow_value = op.CastLike(1.0 / norm_type, weight)
        partial_weight_norm = op.Pow(partial_weight_norm, pow_value)

    max_norm = op.CastLike(op.Constant(value_float=max_norm), weight)
    # This is to avoid weight is zero
    err = op.CastLike(op.Constant(value_float=1e-7), weight)
    partial_weight_norm_ = op.Add(partial_weight_norm, err)
    scales = op.Div(max_norm, partial_weight_norm_)
    partial_weight_renorm = op.Mul(partial_weight, scales)
    # Set values to renormed values where weight_norm > max_norm, but keep the original values where weight_norm <= max_norm
    partial_weight_renorm = op.Where(
        op.Greater(partial_weight_norm, max_norm), partial_weight_renorm, partial_weight
    )
    value = op.ScatterND(weight, op.Unsqueeze(indices, [1]), partial_weight_renorm)
    return value


def aten_embedding_sparse_backward(
    grad: TensorType,
    indices: TensorType,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> TensorType:
    """embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::empty", "aten::empty.memory_format"))
def aten_empty(size: IntType, dtype: int = FLOAT.dtype) -> TTensor:  # type: ignore[type-var]
    # empty(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    # using Zeros to simulate np.empty()
    size = op.Cast(size, to=INT64.dtype)
    zero = op.Constant(value_float=0.0)
    zero = op.Cast(zero, to=dtype)

    return op.Expand(zero, size)


@torch_op("aten::empty_like", trace_only=True)
def aten_empty_like(self: TTensor, dtype: int = -1) -> TTensor:
    """empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    # NOTE: trace_only because both if branches need to be the same type, but we have
    # a cast in the if branch.

    if dtype == -1:
        zero = op.CastLike(0, self)
    else:
        zero = op.Cast(0, to=dtype)

    return _aten_empty_like_onnx(self, zero)


@torch_op("aten::empty_like", private=True)
def _aten_empty_like_onnx(self: TTensor, zero) -> TTensor:
    shape = op.Shape(self)
    return op.Expand(zero, shape)


def aten_empty_quantized(
    size: Sequence[int], qtensor: TensorType, memory_format: Optional[str] = None
) -> TensorType:
    """empty_quantized(int[] size, Tensor qtensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::empty_strided")
def aten_empty_strided(
    size: INT64, stride: INT64  # pylint: disable=unused-argument
) -> TTensor:  # type: ignore[type-var]
    # empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    # using Zeros to simulate empty()
    size = op.Cast(size, to=INT64.dtype)
    zero = op.Constant(value_float=0.0)

    return op.Expand(zero, size)


@torch_op(("aten::eq", "aten::eq.Tensor", "aten::eq.Scalar"))
def aten_eq(self: TTensor, other: TTensor) -> BOOL:
    """eq.Tensor(Tensor self, Tensor other) -> Tensor"""

    return op.Equal(self, other)


@torch_op("aten::equal")
def aten_equal(self: TTensor, other: TTensor) -> BOOL:
    """equal(Tensor self, Tensor other) -> bool"""

    # NOTE: Torch aten::equal returns a single Boolean while ONNX Equal is elementwise.
    # The equivalent Torch op with ONNX Equal is aten::eq.
    elementwise_equal = op.Equal(self, other)
    elementwise_equal_int = op.Cast(elementwise_equal, to=INT64.dtype)
    # ReduceMin does not support bool. So we cast to int64
    all_equal = op.ReduceMin(elementwise_equal_int, keepdims=False)
    return op.Cast(all_equal, to=BOOL.dtype)


def aten_erfinv(self: TensorType) -> TensorType:
    """erfinv(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::exp")
def aten_exp(self: TFloat) -> TFloat:
    """exp(Tensor self) -> Tensor"""

    return op.Exp(self)


@torch_op("aten::exp2")
def aten_exp2(self: TFloat) -> TFloat:
    """exp2(Tensor self) -> Tensor"""

    two = op.Constant(value_int=2)
    two = op.CastLike(two, self)
    return op.Pow(two, self)


@torch_op("aten::expand")
def aten_expand(self: TTensor, size: TInt) -> TTensor:
    """expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)"""
    size = op.Cast(size, to=INT64.dtype)
    # NOTE: PyTorch supports `not changing dim` by -1, but ONNX supports `not changing dim` by 1.
    # To support -1 dim, we need to convert -1 to 1.
    size = op.Abs(size)
    return op.Expand(self, size)


@torch_op("aten::expand_as")
def aten_expand_as(self: TTensor, other: TTensor) -> TTensor:
    """expand_as(Tensor(a) self, Tensor other) -> Tensor(a)"""

    shape = op.Shape(other)
    result = op.Expand(self, shape)
    return result


def aten_expand_copy(self: TensorType, size: INT64, implicit: bool = False) -> TensorType:
    """expand_copy(Tensor self, SymInt[] size, *, bool implicit=False) -> Tensor"""

    raise NotImplementedError()


def aten_expm1(self: TensorType) -> TensorType:
    """expm1(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_eye(n: int) -> TensorType:
    """eye(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


def aten_fake_quantize_per_channel_affine(
    self: TensorType,
    scale: TensorType,
    zero_point: TensorType,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> TensorType:
    """fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor"""

    raise NotImplementedError()


def aten_fake_quantize_per_channel_affine_cachemask(
    self: TensorType,
    scale: TensorType,
    zero_point: TensorType,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> tuple[TensorType, TensorType]:
    """fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output, Tensor mask)"""

    raise NotImplementedError()


def aten_fake_quantize_per_channel_affine_cachemask_backward(
    grad: TensorType, mask: TensorType
) -> TensorType:
    """fake_quantize_per_channel_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor"""

    raise NotImplementedError()


def aten_fake_quantize_per_tensor_affine(
    self: TensorType, scale: float, zero_point: int, quant_min: int, quant_max: int
) -> TensorType:
    """fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor"""

    raise NotImplementedError()


def aten_fake_quantize_per_tensor_affine_cachemask(
    self: TensorType, scale: float, zero_point: int, quant_min: int, quant_max: int
) -> tuple[TensorType, TensorType]:
    """fake_quantize_per_tensor_affine_cachemask(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> (Tensor output, Tensor mask)"""

    raise NotImplementedError()


def aten_fake_quantize_per_tensor_affine_cachemask_backward(
    grad: TensorType, mask: TensorType
) -> TensorType:
    """fake_quantize_per_tensor_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor"""

    raise NotImplementedError()


def aten_fbgemm_linear_fp16_weight(
    input: TensorType, packed_weight: TensorType, bias: TensorType
) -> TensorType:
    """fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor"""

    raise NotImplementedError()


def aten_fbgemm_linear_fp16_weight_fp32_activation(
    input: TensorType, packed_weight: TensorType, bias: TensorType
) -> TensorType:
    """fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor"""

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
    """fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor"""

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
    """fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor"""

    raise NotImplementedError()


def aten_fbgemm_linear_quantize_weight(
    input: TensorType,
) -> tuple[TensorType, TensorType, float, int]:
    """fbgemm_linear_quantize_weight(Tensor input) -> (Tensor, Tensor, float, int)"""

    raise NotImplementedError()


def aten_fbgemm_pack_gemm_matrix_fp16(input: TensorType) -> TensorType:
    """fbgemm_pack_gemm_matrix_fp16(Tensor input) -> Tensor"""

    raise NotImplementedError()


def aten_fbgemm_pack_quantized_matrix(input: TensorType) -> TensorType:
    """fbgemm_pack_quantized_matrix(Tensor input) -> Tensor"""

    raise NotImplementedError()


def aten_feature_alpha_dropout(input: TensorType, p: float, train: bool) -> TensorType:
    """feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor"""

    raise NotImplementedError()


def aten_feature_dropout(input: TensorType, p: float, train: bool) -> TensorType:
    """feature_dropout(Tensor input, float p, bool train) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::fill", "aten::fill.Tensor"))
def aten_fill(self: TTensor, value: TTensor) -> TTensor:
    """fill.Tensor(Tensor self, Tensor value) -> Tensor"""

    # after fill, the self Tensor should keep origianl type
    shape = op.Shape(self)
    expanded = op.Expand(value, shape)
    result = op.CastLike(expanded, self)
    return result


def aten_fix(self: TensorType) -> TensorType:
    """fix(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::flip")
def aten_flip(self: TTensor, dims: INT64) -> TTensor:
    """flip(Tensor self, int[] dims) -> Tensor"""

    shape_dim = op.Shape(dims)
    neg_1 = op.Constant(value_int=-1)
    starts = op.Expand(neg_1, shape_dim)  # something like [-1, -1, -1]
    steps = op.Expand(neg_1, shape_dim)  # something like [-1, -1, -1]
    ends = op.Expand(_INT64_MIN, shape_dim)  # something like [-xxx, -xxx, -xxx]
    result = op.Slice(self, starts, ends, dims, steps)
    return result


def aten_fliplr(self: TensorType) -> TensorType:
    """fliplr(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_flipud(self: TensorType) -> TensorType:
    """flipud(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::floor")
def aten_floor(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """floor(Tensor self) -> Tensor"""

    return op.Floor(self)


@torch_op("aten::floor_divide")
def aten_floor_divide(self: TFloat, other: TFloat) -> TFloat:
    """floor_divide(Tensor self, Tensor other) -> Tensor"""

    return op.Floor(op.Div(self, other))


def aten_fmax(self: TensorType, other: TensorType) -> TensorType:
    """fmax(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_fmin(self: TensorType, other: TensorType) -> TensorType:
    """fmin(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::fmod")
def aten_fmod(self: TRealOrUInt8, other: TRealOrUInt8) -> TRealOrUInt8:
    """fmod.Tensor(Tensor self, Tensor other) -> Tensor"""

    return op.Mod(self, other, fmod=1)


@torch_op("aten::frac")
def aten_frac(self: TFloat) -> TFloat:
    """frac(Tensor self) -> Tensor

    Computes the fractional portion of each element in input.
    """

    # https://pytorch.org/docs/stable/generated/torch.frac.html
    return op.Sub(self, op.Mul(op.Floor(op.Abs(self)), op.Sign(self)))


def aten_frexp(self: TensorType) -> tuple[TensorType, TensorType]:
    """frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)"""

    raise NotImplementedError()


def aten_frobenius_norm(self: TensorType) -> TensorType:
    """frobenius_norm(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_from_file(
    filename: str, shared: Optional[bool] = None, size: Optional[int] = 0
) -> TensorType:
    """from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::full")
def aten_full(size: INT64, fill_value: FLOAT, dtype: int = FLOAT.dtype):
    """full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    size = op.Cast(size, to=INT64.dtype)
    fill_value = op.Cast(fill_value, to=dtype)
    return op.Expand(fill_value, size)


@torch_op("aten::full_like")
def aten_full_like(self, fill_value: TTensor) -> TTensor:
    """full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    fill_value = op.CastLike(fill_value, self)
    self_shape = op.Shape(self)

    return op.Expand(fill_value, self_shape)


@torch_op("aten::full_like")
def aten_full_like_dtype(self, fill_value: TTensor, dtype: int) -> TTensor:
    """full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    fill_value = op.Cast(fill_value, to=dtype)
    self_shape = op.Shape(self)

    return op.Expand(fill_value, self_shape)


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
    """fused_moving_avg_obs_fake_quant(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::gather")
def aten_gather(
    self: TReal,
    dim: int,
    index: TInt,
    sparse_grad: bool = False,  # pylint: disable=unused-argument
) -> TReal:
    """gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor"""

    if op.Size(op.Shape(index)) == 0:  # When (index) is empty, return (self)
        result = self
    else:
        if op.Size(op.Shape(self)) == 0:  # Unsqueeze for GatherElements op
            self = op.Reshape(self, op.Constant(value_ints=[-1]))
        if op.Size(index) == 0:  # Return empty array
            result = op.CastLike(index, self)
        else:
            index = op.Cast(index, to=INT64.dtype)
            result = op.GatherElements(self, index, axis=dim)
    return result


def aten_gather_backward(
    grad: TensorType, self: TensorType, dim: int, index: TensorType, sparse_grad: bool
) -> TensorType:
    """gather_backward(Tensor grad, Tensor self, int dim, Tensor index, bool sparse_grad) -> Tensor"""

    raise NotImplementedError()


def aten_gcd(self: TensorType, other: TensorType) -> TensorType:
    """gcd(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::ge", "aten::ge.Tensor", "aten::ge.Scalar", "aten::greater_equal"))
def aten_ge(self: TReal, other: TReal) -> BOOL:
    """ge.Tensor(Tensor self, Tensor other) -> Tensor"""

    return op.GreaterOrEqual(self, other)


@torch_op(("aten::ge", "aten::ge.Tensor", "aten::ge.Scalar", "aten::greater_equal"))
def aten_ge_bool(self: BOOL, other: BOOL) -> BOOL:
    """ge.Tensor(Tensor self, Tensor other) -> Tensor"""

    # self, other, self >= other
    #    F,    F,    T
    #    F,    T,    F
    #    T,    F,    T
    #    T,    T,    T

    return op.Or(self, op.Not(other))


def aten_geqrf(self: TensorType) -> tuple[TensorType, TensorType]:
    """geqrf(Tensor self) -> (Tensor a, Tensor tau)"""

    raise NotImplementedError()


def aten_ger(self: TensorType, vec2: TensorType) -> TensorType:
    """ger(Tensor self, Tensor vec2) -> Tensor"""

    raise NotImplementedError()


# NOTE: The name is made up for `getitem` to be included in the registry
@torch_op("aten::getitem")
def aten_getitem(self: Sequence[TTensor], i: INT64) -> TTensor:
    return op.SequenceAt(self, i)


@torch_op("aten::grid_sampler", trace_only=True)
def aten_grid_sampler(
    input: TTensor,
    grid: TTensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TTensor:
    """grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor"""

    inter_mode_options = ("bilinear", "nearest", "bicubic")
    inter_mode_str = inter_mode_options[interpolation_mode]

    padding_mode_options = ("zeros", "border", "reflection")
    padding_mode_str = padding_mode_options[padding_mode]

    # Only one onnx Op so don't put into private function
    return op.GridSample(
        input,
        grid,
        align_corners=align_corners,
        mode=inter_mode_str,
        padding_mode=padding_mode_str,
    )


@torch_op("aten::grid_sampler_2d", trace_only=True)
def aten_grid_sampler_2d(
    input: TTensor,
    grid: TTensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TTensor:
    """grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor"""

    inter_mode_options = ("bilinear", "nearest", "bicubic")
    inter_mode_str = inter_mode_options[interpolation_mode]

    padding_mode_options = ("zeros", "border", "reflection")
    padding_mode_str = padding_mode_options[padding_mode]

    # Only one onnx Op so don't put into private function
    return op.GridSample(
        input,
        grid,
        align_corners=align_corners,
        mode=inter_mode_str,
        padding_mode=padding_mode_str,
    )


def aten_grid_sampler_2d_backward(
    grad_output: TensorType,
    input: TensorType,
    grid: TensorType,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask: Sequence[bool],
) -> tuple[TensorType, TensorType]:
    """grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_grid_sampler_3d(
    input: TensorType,
    grid: TensorType,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TensorType:
    """grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor"""

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
    """grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_group_norm(
    input: TensorType,
    num_groups: int,
    weight: Optional[TensorType] = None,
    bias: Optional[TensorType] = None,
    eps: float = 1e-05,
    cudnn_enabled: bool = True,
) -> TensorType:
    """group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor"""

    raise NotImplementedError()


def aten_gru_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> TensorType:
    """gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::gt", "aten::gt.Scalar", "aten::greater"))
def aten_gt(self: TReal, other: TReal) -> BOOL:
    """gt.Tensor(Tensor self, Tensor other) -> Tensor"""

    return op.Greater(self, other)


@torch_op(("aten::gt", "aten::gt.Scalar", "aten::greater"))
def aten_gt_bool(self: BOOL, other: BOOL) -> BOOL:
    """gt.Tensor(Tensor self, Tensor other) -> Tensor"""
    # self, other, self > other
    #    F,    F,    F
    #    F,    T,    F
    #    T,    F,    T
    #    T,    T,    F

    return op.And(self, op.Not(other))


def aten_hamming_window(window_length: int) -> TensorType:
    """hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


def aten_hann_window(window_length: int) -> TensorType:
    """hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


def aten_hardshrink(self: TensorType, lambd: float = 0.5) -> TensorType:
    """hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor"""

    raise NotImplementedError()


def aten_hardshrink_backward(
    grad_out: TensorType, self: TensorType, lambd: float
) -> TensorType:
    """hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::heaviside")
def aten_heaviside(self: TReal, values: TReal) -> TReal:
    """heaviside(Tensor self, Tensor values) -> Tensor"""

    zero = op.CastLike(0, self)
    one = op.CastLike(1, self)
    intermediate = op.Where(op.Less(self, zero), zero, one)

    return op.Where(op.Equal(self, zero), values, intermediate)


def aten_hinge_embedding_loss(
    self: TensorType, target: TensorType, margin: float = 1.0, reduction: int = 1
) -> TensorType:
    """hinge_embedding_loss(Tensor self, Tensor target, float margin=1.0, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


def aten_histc(
    self: TensorType, bins: int = 100, min: float = 0.0, max: float = 0.0
) -> TensorType:
    """histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor"""

    raise NotImplementedError()


def aten_histogramdd(
    self: TensorType,
    bins: Sequence[int],
    range: Optional[float] = None,
    weight: Optional[TensorType] = None,
    density: bool = False,
) -> tuple[TensorType, TensorType]:
    """histogramdd(Tensor self, int[] bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)"""

    raise NotImplementedError()


def aten_hspmm(mat1: TensorType, mat2: TensorType) -> TensorType:
    """hspmm(Tensor mat1, Tensor mat2) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::hstack")
def aten_hstack(tensors: Sequence[TTensor]) -> TTensor:
    """hstack(Tensor[] tensors) -> Tensor"""

    @graph()
    def reshape_to_atleast_2d(tensor):
        shape = op.Shape(tensor)
        rank = op.Size(shape)
        if rank <= 1:
            tensor = op.Reshape(tensor, op.Constant(value_ints=[1, -1]))
        return tensor

    tensors_atleast_2d = op.SequenceMap(tensors, body=reshape_to_atleast_2d)

    result = op.ConcatFromSequence(tensors_atleast_2d, axis=1, new_axis=0)

    # hstack expects a non-empty sequence of tensors. So we don't need to check for length
    rank_1d_or_less = op.Less(op.Size(op.Shape(op.SequenceAt(tensors, 0))), 2)
    if rank_1d_or_less:
        result = op.Reshape(result, op.Constant(value_ints=[-1]))
    return result


def aten_hypot(self: TensorType, other: TensorType) -> TensorType:
    """hypot(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_i0(self: TensorType) -> TensorType:
    """i0(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_igamma(self: TensorType, other: TensorType) -> TensorType:
    """igamma(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_igammac(self: TensorType, other: TensorType) -> TensorType:
    """igammac(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_imag(self: TensorType) -> TensorType:
    """imag(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


def _are_consecutive(sorted_list: Sequence[int]) -> bool:
    """Returns True if a sorted list contains consecutive numbers."""
    if not sorted_list:
        return True

    return sorted_list == list(range(min(sorted_list), max(sorted_list) + 1))


def _has_none_in_middle(indices) -> bool:
    """Returns True if there is a None in the middle of the list."""
    not_none_indices = [i for i, idx in enumerate(indices) if idx is not None]
    return not _are_consecutive(not_none_indices)


def _shape_of_broadcast_tensors(*args: TensorType) -> INT64:
    """Returns the broadcasted shape of the given tensors."""
    broadcasted = op.Max(*args)
    return op.Shape(broadcasted)


@torch_op(("aten::index.Tensor", "aten::_unsafe_index.Tensor"), trace_only=True)
def aten_index(self: TensorType, indices: Sequence[Optional[INT64]]) -> TensorType:
    """index.Tensor(Tensor self, Tensor?[] indices) -> Tensor

    NOTE: Understanding `aten::index`
    For `arg0` with shape `[7, 3, 4, 5, 6]`
    The indexing operation `arg0[0, :, 1:2, tensor([[4,5]])]` will be translated to

    ```
    +>  select: i64[3, 4, 5, 6] = torch.ops.aten.select.int(arg0, 0, 0);
    +>  slice_1: i64[3, 4, 5, 6] = torch.ops.aten.slice.Tensor(select, 0, 0, 9223372036854775807);
    +>  slice_2: i64[3, 1, 5, 6] = torch.ops.aten.slice.Tensor(slice_1, 1, 1, 2);
    +>  index: i64[3, 1, 1, 2, 6] = torch.ops.aten.index.Tensor(slice_2, [None, None, arg1]);
    ```

    Here,
    - `indices = [None, None, arg1]` is equivalent to `indices = [None, None, arg1, None]`
    - The operation `arg0[0, :, 1:2, tensor([[4,5]])]` is equivalent to `arg0[0, :, 1:2, tensor([[4,5]]), :]`

    None in `indices` are like fillers for dimensions that cannot be removed in the process.
    """

    self_rank = len(self.shape)
    index_ranks = [len(index.shape) for index in indices if index is not None]
    advanced_indexing_rank = max(index_ranks)

    # reordered_positions is the permutation of the index positions where
    # positions with None are move to the end of the list
    # For example, if indices = [None, 1, None, 2], then reordered_positions = [1, 3, 0, 2]
    reordered_positions = sorted(range(len(indices)), key=lambda i: (indices[i] is None, i))
    # Fill the list with the remaining indices up to the rank of the tensor self.
    # For example, if indices = [None, 1, None, 2], and the rank of self is 6,
    # then reordered_positions = [1, 3, 0, 2, 4, 5]
    reordered_positions = [
        *reordered_positions,
        *range(len(reordered_positions), self_rank),
    ]
    # Transpose self according to the reordered positions
    self = op.Transpose(self, perm=reordered_positions)

    # Broadcast the indices to the same shape then concatenate
    not_none_indices = [idx for idx in indices if idx is not None]
    broadcast_shape = _shape_of_broadcast_tensors(*not_none_indices)
    final_index = op.Concat(
        *(op.Unsqueeze(op.Expand(idx, broadcast_shape), -1) for idx in not_none_indices),
        axis=-1,
    )

    self = op.GatherND(self, final_index, batch_dims=0)

    if _has_none_in_middle(indices):
        # If there is None in the middle, Advanced Indexing cannot decide where to put
        # the new dimensions. So it places them in the front, like GatherND does.
        return self

    # When the indices are consecutive, Advanced Indexing will place the new dimensions
    # (aka. the broadcasted shape) in the middle, replacing the original [x1, ..., xk] axes.
    #
    # Input index axes (three parts):
    #   [
    #      x_None_front_1, ... x_None_front_m,
    #      x1, ..., xk,
    #      x_None_back_1, ..., x_None_back_m
    #   ]
    # GatherND result axes:
    #   [
    #      *broadcasted_shape(x1, x2, ..., xk),
    #      x_None_front_1, ... x_None_front_m,
    #      x_None_back_1, ..., x_None_back_m
    #   ]
    # (Transpose here)
    # Advanced indexing result axes:
    #   [
    #      x_None_front_1, ... x_None_front_m,
    #      *brocasted_shape(x1, x2, ..., xk),
    #      x_None_back_1, ..., x_None_back_m
    #   ]
    #
    # Need to transpose the result of GatherND to match this axes ordering.
    first_not_none_position = reordered_positions[0]  # x_None_front_m + 1
    starting_position_of_none_in_back = (
        advanced_indexing_rank + first_not_none_position
    )  # x_None_back_1
    result_rank = self_rank - len(not_none_indices) + advanced_indexing_rank
    perm = [
        *range(
            advanced_indexing_rank, starting_position_of_none_in_back
        ),  # None_front_1...x_None_back_1
        *range(0, advanced_indexing_rank),  # 0...len(broadcasted_shape)
        *range(
            starting_position_of_none_in_back,
            result_rank,
        ),  # None_back_1...None_back_m
    ]

    return op.Transpose(self, perm=perm)


def aten_index_add(
    self: TensorType, dim: int, index: TensorType, source: TensorType, alpha: float = 1
) -> TensorType:
    """index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor"""

    raise NotImplementedError()


def aten_index_copy(
    self: TensorType, dim: int, index: TensorType, source: TensorType
) -> TensorType:
    """index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::index_put")
def aten_index_put(
    self: TReal,
    indices: Sequence[INT64],
    values: TReal,
    accumulate: bool = False,
) -> TReal:
    """index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"""

    index = op.SequenceAt(indices, 0)  # assume indices only have 1 element
    # change array([1,3]) to array([[1,1,1,1,1],[3,3,3,3,3]])
    self_dim_1 = op.Gather(op.Shape(self), 1)
    index_dim_0 = op.Gather(op.Shape(index), 0)
    neg_1 = op.Constant(value_ints=[-1])
    shape = op.Concat(op.Reshape(self_dim_1, neg_1), op.Reshape(index_dim_0, neg_1), axis=0)
    new_ind = op.Expand(index, shape)
    new_ind_t = op.Transpose(new_ind)

    if op.Cast(accumulate, to=BOOL.dtype):
        # put values into zeros array first, then add to input
        zeros = op.Expand(op.Constant(value_float=0.0), op.Shape(self))
        zeros = op.CastLike(zeros, values)
        result = op.ScatterElements(zeros, new_ind_t, values)
        # FIXME: type promotion
        result = op.CastLike(result, self)
        result = op.Add(result, self)
    else:
        result = op.ScatterElements(self, new_ind_t, values)
    return result


@torch_op("aten::index_put")
def aten_index_put_bool(
    self: TReal,
    indices: Sequence[BOOL],
    values: TReal,
    accumulate: bool = False,
) -> TReal:
    """index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"""

    index = op.SequenceAt(indices, 0)  # assume indices only have 1 element
    # FIXME: ORT ArgMax fails on INT64 input even though ONNX allows it
    index_int = op.Cast(index, to=INT32.dtype)
    # if all False, return self
    if op.ReduceSum(index_int) == 0:
        result = self
    else:
        # change array([F,F,T,F,F]) to array([2])
        index = op.ArgMax(index_int)  # assume index only have 1 True
        # change array([2]) to array([2,2,2,2,2])
        self_dim_1 = op.Gather(op.Shape(self), 1)
        index_dim_0 = op.Gather(op.Shape(index), 0)
        neg_1 = op.Constant(value_ints=[-1])
        shape = op.Concat(
            op.Reshape(self_dim_1, neg_1), op.Reshape(index_dim_0, neg_1), axis=0
        )
        new_ind = op.Expand(index, shape)
        new_ind_t = op.Transpose(new_ind)

        # values must have same rank with input(self)
        if op.Size(op.Shape(values)) < op.Size(op.Shape(self)):  # type: ignore[operator]
            values = op.Unsqueeze(values, op.Constant(value_ints=[0]))

        if op.Cast(accumulate, to=BOOL.dtype):
            zeros = op.Expand(op.Constant(value_float=0.0), op.Shape(self))
            zeros = op.CastLike(zeros, values)
            result = op.ScatterElements(zeros, new_ind_t, values)
            # FIXME: type promotion
            result = op.CastLike(result, self)
            result = op.Add(result, self)
        else:
            result = op.ScatterElements(self, new_ind_t, values)

    return result


def aten_index_reduce(
    self: TensorType,
    dim: int,
    index: TensorType,
    source: TensorType,
    reduce: str,
    include_self: bool = True,
) -> TensorType:
    """index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::index_select")
def aten_index_select(self: TTensor, dim: int, index: IntType) -> TTensor:
    """index_select(Tensor self, int dim, Tensor index) -> Tensor"""

    self_is_scalar = op.Size(op.Shape(self)) == 0
    if self_is_scalar:
        self = op.Reshape(self, op.Constant(value_ints=[-1]))

    # Index may be a scalar. Reshape it to a rank 1 tensor.
    index = op.Reshape(index, op.Constant(value_ints=[-1]))
    index = op.Cast(index, to=INT64.dtype)
    result = op.Gather(self, index, axis=dim)

    if self_is_scalar:
        result = op.Squeeze(result)

    return result


def aten_index_select_backward(
    grad: TensorType, self_sizes: INT64, dim: int, index: TensorType
) -> TensorType:
    """index_select_backward(Tensor grad, SymInt[] self_sizes, int dim, Tensor index) -> Tensor"""

    raise NotImplementedError()


def aten_indices(self: TensorType) -> TensorType:
    """indices(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


def aten_indices_copy(self: TensorType) -> TensorType:
    """indices_copy(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_inner(self: TensorType, other: TensorType) -> TensorType:
    """inner(Tensor self, Tensor other) -> Tensor"""

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
    """instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor"""

    raise NotImplementedError()


def aten_int_repr(self: TensorType) -> TensorType:
    """int_repr(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_inverse(self: TensorType) -> TensorType:
    """inverse(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_is_coalesced(self: TensorType) -> bool:
    """is_coalesced(Tensor self) -> bool"""

    raise NotImplementedError()


def aten_is_complex(self: TensorType) -> bool:
    """is_complex(Tensor self) -> bool"""

    raise NotImplementedError()


def aten_is_conj(self: TensorType) -> bool:
    """is_conj(Tensor self) -> bool"""

    raise NotImplementedError()


def aten_is_distributed(self: TensorType) -> bool:
    """is_distributed(Tensor self) -> bool"""

    raise NotImplementedError()


def aten_is_floating_point(self: TensorType) -> bool:
    """is_floating_point(Tensor self) -> bool"""

    raise NotImplementedError()


def aten_is_inference(self: TensorType) -> bool:
    """is_inference(Tensor self) -> bool"""

    raise NotImplementedError()


def aten_is_leaf(self: TensorType) -> bool:
    """is_leaf(Tensor self) -> bool"""

    raise NotImplementedError()


def aten_is_neg(self: TensorType) -> bool:
    """is_neg(Tensor self) -> bool"""

    raise NotImplementedError()


@torch_op("aten::is_nonzero")
def aten_is_nonzero(self: Union[RealType, BOOL]) -> BOOL:
    """is_nonzero(Tensor self) -> bool"""

    # if size != 1, return False
    # else [0],[True],[0.0] return True, others return False
    result = op.Not(op.Size(self) != 1)
    if result:
        result = op.Cast(self, to=BOOL.dtype)
    return result


def aten_is_pinned(self: TensorType, device: Optional[str] = None) -> bool:
    """is_pinned(Tensor self, Device? device=None) -> bool"""

    raise NotImplementedError()


@torch_op("aten::is_same_size")
def aten_is_same_size(self: TTensor, other: TTensor) -> BOOL:
    """is_same_size(Tensor self, Tensor other) -> bool"""

    # Cannot compare different shape of two tensors using op.Equal()
    # So we need to compare the rank first, if rank is same, then compare shape
    self_rank = op.Size(op.Shape(self))
    other_rank = op.Size(op.Shape(other))
    result = op.Equal(self_rank, other_rank)
    if result:  # Same rank, then compare shape
        self_shape = op.Shape(self)
        other_shape = op.Shape(other)
        result_bool = op.Equal(self_shape, other_shape)
        result_int = op.Cast(result_bool, to=INT8.dtype)
        result = op.Cast(op.ReduceMin(result_int, keepdims=False), to=BOOL.dtype)

    return result


def aten_is_set_to(self: TensorType, tensor: TensorType) -> bool:
    """is_set_to(Tensor self, Tensor tensor) -> bool"""

    raise NotImplementedError()


def aten_is_signed(self: TensorType) -> bool:
    """is_signed(Tensor self) -> bool"""

    raise NotImplementedError()


def aten_is_vulkan_available() -> bool:
    """is_vulkan_available() -> bool"""

    raise NotImplementedError()


@torch_op("aten::isclose")
def aten_isclose(
    self: TReal,
    other: TReal,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,  # pylint: disable=unused-argument
) -> BOOL:
    """isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor"""

    # FIXME: check equal_nan when self and other are all NaN
    # |input - other| <= atol + rtol x |other|
    left_part = op.Abs(op.Sub(self, other))
    right_part = op.Add(atol, op.Mul(rtol, op.Abs(other)))
    result = op.LessOrEqual(left_part, right_part)
    return result


@torch_op("aten::isfinite")
def aten_isfinite(self: TFloatHighPrecision) -> BOOL:
    """isfinite(Tensor self) -> Tensor"""

    # IsInf only supports FLOAT and DOUBLE
    not_inf = op.Not(op.IsInf(self))
    not_nan = op.Not(op.IsNaN(self))  # TODO: The test case doesnt cover this condition
    return op.And(not_inf, not_nan)


@torch_op("aten::isinf")
def aten_isinf(self: TFloatOrBFloat16) -> BOOL:
    """isinf(Tensor self) -> Tensor"""

    # Added Cast inside the function so it can support all real dtypes naturally
    self = op.Cast(self, to=FLOAT.dtype)
    return op.IsInf(self)


@torch_op("aten::isnan")
def aten_isnan(self: TFloatOrBFloat16) -> BOOL:
    """isnan(Tensor self) -> Tensor"""

    return op.IsNaN(self)


@torch_op("aten::isneginf")
def aten_isneginf(self: TFloatOrBFloat16) -> BOOL:
    """isneginf(Tensor self) -> Tensor"""

    # Added Cast inside the function so it can support all real dtypes naturally
    self = op.Cast(self, to=FLOAT.dtype)
    return op.And(op.Less(self, 0), op.IsInf(self))


@torch_op("aten::isposinf")
def aten_isposinf(self: TFloatOrBFloat16) -> BOOL:
    """isposinf(Tensor self) -> Tensor"""

    # Added Cast inside the function so it can support all real dtypes naturally
    self = op.Cast(self, to=FLOAT.dtype)
    return op.And(op.Greater(self, 0), op.IsInf(self))


def aten_isreal(self: TensorType) -> TensorType:
    """isreal(Tensor self) -> Tensor"""

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
    """istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False) -> Tensor"""

    raise NotImplementedError()


def aten_item(self: TensorType) -> float:
    """item(Tensor self) -> Scalar"""

    raise NotImplementedError()


def aten_kaiser_window(window_length: int) -> TensorType:
    """kaiser_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


def aten_kl_div(
    self: TensorType, target: TensorType, reduction: int = 1, log_target: bool = False
) -> TensorType:
    """kl_div(Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor"""

    raise NotImplementedError()


def aten_kron(self: TensorType, other: TensorType) -> TensorType:
    """kron(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_kthvalue(
    self: TensorType, k: int, dim: int = -1, keepdim: bool = False
) -> tuple[TensorType, TensorType]:
    """kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)"""

    raise NotImplementedError()


@torch_op("aten::layer_norm", trace_only=True)
def aten_layer_norm(
    input: TReal,
    normalized_shape: INT64,
    weight: Optional[TReal] = None,
    bias: Optional[TReal] = None,
    eps: float = 1e-05,
) -> TReal:
    """layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor"""

    # trace_only to use Python to obtain start_axis
    start_axis = -len(normalized_shape)

    if weight is None:
        one = op.Constant(value_float=1.0)
        weight = op.Expand(one, op.Shape(input, start=start_axis))

    if bias is None:
        zero = op.Constant(value_float=0.0)
        bias = op.Expand(zero, op.Shape(input, start=start_axis))

    return _aten_layer_norm_onnx(input, weight, bias, axis=start_axis, eps=eps)


@torch_op("aten::layer_norm", private=True)
def _aten_layer_norm_onnx(
    input: TReal,
    weight: TReal,
    bias: TReal,
    axis: int,
    eps: float = 1e-05,
) -> TReal:
    """layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor"""

    # TODO(justinchuby): Use OptionalHasElement after onnx/onnx#4982
    result, _, _ = op.LayerNormalization(input, weight, bias, axis=axis, epsilon=eps)
    return result


def aten_lcm(self: TensorType, other: TensorType) -> TensorType:
    """lcm(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_ldexp(self: TensorType, other: TensorType) -> TensorType:
    """ldexp.Tensor(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::le", "aten::le.Tensor"))
def aten_le(self: TReal, other: TReal) -> BOOL:
    """le.Tensor(Tensor self, Tensor other) -> Tensor"""

    return op.LessOrEqual(self, other)


@torch_op(("aten::le", "aten::le.Tensor", "aten::less_equal"))
def aten_le_bool(self: BOOL, other: BOOL) -> BOOL:
    """le.Tensor(Tensor self, Tensor other) -> Tensor"""

    # self, other, self <= other
    #    F,    F,    T
    #    F,    T,    T
    #    T,    F,    F
    #    T,    T,    T

    return op.Or(other, op.Not(self))


def aten_lerp(self: TensorType, end: TensorType, weight: TensorType) -> TensorType:
    """lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor"""

    raise NotImplementedError()


def aten_lgamma(self: TensorType) -> TensorType:
    """lgamma(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_lift(self: TensorType) -> TensorType:
    """lift(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_lift_fresh(self: TensorType) -> TensorType:
    """lift_fresh(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


@torch_op("aten::lift_fresh_copy")
def aten_lift_fresh_copy(self: TensorType) -> TensorType:
    """lift_fresh_copy(Tensor self) -> Tensor"""

    return op.Identity(self)


def aten_linear_backward(
    self: TensorType, grad_output: TensorType, weight: TensorType, output_mask: Sequence[bool]
) -> tuple[TensorType, TensorType, TensorType]:
    """linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""

    raise NotImplementedError()


@torch_op("aten::linspace", trace_only=True)
def aten_linspace(
    start: TRealUnlessFloat16OrInt8, end: TRealUnlessFloat16OrInt8, steps: int, dtype: int = -1
) -> TensorType:
    """linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    if dtype == -1:
        zero = op.CastLike(0.0, steps)
        one = op.CastLike(1.0, steps)
    elif _range_supported(dtype):
        zero = op.Cast(0, to=dtype)
        one = op.Cast(1, to=dtype)
        start = op.Cast(start, to=dtype)
        end = op.Cast(end, to=dtype)
        steps = op.Cast(steps, to=dtype)
    else:
        # Cast input to float if dtype is not supported by Range,
        # because the input dtype may be e.g. bfloat16 / int8 etc.
        # which Range does not support. The output type is ensured because the output
        # is casted to the specified dtype.
        zero = op.Cast(0.0, to=FLOAT.dtype)
        one = op.Cast(1.0, to=FLOAT.dtype)
        start = op.Cast(start, to=FLOAT.dtype)
        end = op.Cast(end, to=FLOAT.dtype)
        steps = op.Cast(steps, to=FLOAT.dtype)

    range_tensor = op.Range(zero, steps, one)

    start = op.CastLike(start, end)
    step = op.Div(
        op.Sub(end, start),
        op.Sub(steps, one),
    )

    return op.Add(op.Mul(range_tensor, step), start)


@torch_op("aten::log")
def aten_log(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """log(Tensor self) -> Tensor"""

    return op.Log(self)


@torch_op("aten::log10")
def aten_log10(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """log10(Tensor self) -> Tensor"""

    return op.Div(op.Log(self), op.CastLike(op.Log(10.0), self))


@torch_op("aten::log1p")
def aten_log1p(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """log1p(Tensor self) -> Tensor"""

    return op.Log(op.Add(self, 1.0))


@torch_op("aten::log2")
def aten_log2(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """log2(Tensor self) -> Tensor"""

    return op.Div(op.Log(self), op.CastLike(op.Log(2.0), self))


@torch_op("aten::logaddexp")
def aten_logaddexp(self: TFloatOrBFloat16, other: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """logaddexp(Tensor self, Tensor other) -> Tensor"""

    return op.Log(op.Add(op.Exp(self), op.Exp(other)))


@torch_op("aten::logaddexp2")
def aten_logaddexp2(self: TFloatOrBFloat16, other: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """logaddexp2(Tensor self, Tensor other) -> Tensor"""
    summation = op.Add(op.Pow(2.0, self), op.Pow(2.0, other))

    return op.Div(op.Log(summation), op.CastLike(op.Log(2.0), self))


@torch_op("aten::logcumsumexp")
def aten_logcumsumexp(self: TFloatOrBFloat16, dim: int) -> TFloatOrBFloat16:
    """logcumsumexp(Tensor self, int dim) -> Tensor"""

    self_rank = op.Size(op.Shape(self))
    if self_rank == 0:
        result = self
    else:
        # Make dim 1-d
        dims = op.Unsqueeze(dim, axes=[0])
        # This uses the max trick to avoid overflow:
        # Assuming A = [a_1, a_2, ..., a_n] and the output
        # out = [out_1, out_2, ..., out_n], then
        # out_i = log(cumsum(exp(A)))_i
        #       = log(exp(a_1) + ... + exp(a_i))
        #       = log(exp(a_1) + ... + exp(a_i)) - max(A) + max(A)
        #       = log((exp(a_1) + ... + exp(a_i)) / exp(max(A))) + max(A)
        #       = log(exp(a_1-max(A)) + ... + exp(a_i-max(A))) + max(A)
        #       = log(sum<j=1...i>(exp(a_j - max(A)))) + max(A)
        # Vectorizing for all i, we get the expression below.
        self_max = op.ReduceMax(self, dims)
        result = op.Log(op.CumSum(op.Exp(self - self_max), dims)) + self_max
    return result


@torch_op("aten::logdet")
def aten_logdet(self: TFloat) -> TFloat:
    """logdet(Tensor self) -> Tensor"""

    return op.Log(op.Det(self))


@torch_op(
    (
        "aten::logical_and",
        "aten::bitwise_and",
        "aten::bitwise_and.Tensor",
        "aten::bitwise_and.Scalar",
        "aten::bitwise_and.Scalar_Tensor",
    )
)
def aten_logical_and(self: BOOL, other: BOOL) -> BOOL:
    """logical_and(Tensor self, Tensor other) -> Tensor"""

    return op.And(self, other)


@torch_op(("aten::logical_not", "aten::bitwise_not"))
def aten_logical_not(self: BOOL) -> BOOL:
    """logical_not(Tensor self) -> Tensor"""

    return op.Not(self)


@torch_op(
    (
        "aten::logical_or",
        "aten::bitwise_or",
        "aten::bitwise_or.Tensor",
        "aten::bitwise_or.Scalar",
        "aten::bitwise_or.Scalar_Tensor",
        "aten::add",
        "aten::add.Tensor",
    )
)
def aten_logical_or(self: BOOL, other: BOOL) -> BOOL:
    """logical_or(Tensor self, Tensor other) -> Tensor"""

    return op.Or(self, other)


@torch_op(
    (
        "aten::logical_xor",
        "aten::bitwise_xor",
        "aten::bitwise_xor.Tensor",
        "aten::bitwise_xor.Scalar",
        "aten::bitwise_xor.Scalar_Tensor",
    )
)
def aten_logical_xor(self: BOOL, other: BOOL) -> BOOL:
    """logical_xor(Tensor self, Tensor other) -> Tensor"""

    return op.Xor(self, other)


@torch_op("aten::logit", private=True)
def _aten_logit_onnx(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    return op.Log(op.Div(self, op.Sub(1.0, self)))


@torch_op("aten::logit", private=True)
def _aten_logit_clamp_onnx(self: TFloatOrBFloat16, eps: float) -> TFloatOrBFloat16:
    temporary_self = op.Where(self <= 1.0 - eps, self, 1.0 - eps)
    z = op.Where(temporary_self < eps, eps, temporary_self)

    return op.Log(op.Div(z, op.Sub(1.0, z)))


@torch_op("aten::logit", trace_only=True)
def aten_logit(self: TFloatOrBFloat16, eps: Optional[float] = None) -> TFloatOrBFloat16:
    """logit(Tensor self, float? eps=None) -> Tensor"""
    if eps is None:
        return _aten_logit_onnx(self)
    return _aten_logit_clamp_onnx(self, eps)


def aten_logspace(start: float, end: float, steps: int, base: float = 10.0) -> TensorType:
    """logspace(Scalar start, Scalar end, int steps, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::logsumexp")
def aten_logsumexp(self: TFloat, dim: INT64, keepdim: int = False) -> TFloat:
    """logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor"""

    if op.Size(op.Shape(self)) == 0:
        # A scalar
        result = self
    else:
        result = op.ReduceLogSumExp(self, dim, keepdims=keepdim)
    return result


def aten_lshift(self: TensorType, other: TensorType) -> TensorType:
    """__lshift__.Tensor(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_lstm_cell(
    input: TensorType,
    hx: Sequence[TensorType],
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> tuple[TensorType, TensorType]:
    """lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)"""

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
    """lstm_mps_backward(Tensor grad_y, Tensor? grad_hy, Tensor? grad_cy, Tensor z_state, Tensor cell_state_fwd, Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor[], Tensor[])"""

    raise NotImplementedError()


@torch_op(("aten::lt", "aten::lt.Scalar", "aten::less"))
def aten_lt(self: TReal, other: TReal) -> BOOL:
    """lt.Tensor(Tensor self, Tensor other) -> Tensor"""

    return op.Less(self, other)


@torch_op(("aten::lt", "aten::lt.Scalar", "aten::less"))
def aten_lt_bool(self: BOOL, other: BOOL) -> BOOL:
    """lt.Tensor(Tensor self, Tensor other) -> Tensor"""

    # self, other, self < other
    #    F,    F,    F
    #    F,    T,    T
    #    T,    F,    F
    #    T,    T,    F

    return op.And(other, op.Not(self))


def aten_lu_solve(self: TensorType, LU_data: TensorType, LU_pivots: TensorType) -> TensorType:
    """lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor"""

    raise NotImplementedError()


def aten_lu_unpack(
    LU_data: TensorType,
    LU_pivots: TensorType,
    unpack_data: bool = True,
    unpack_pivots: bool = True,
) -> tuple[TensorType, TensorType, TensorType]:
    """lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)"""

    raise NotImplementedError()


@torch_op("aten::mH")
def aten_mH(self: TRealOrUInt8) -> TRealOrUInt8:
    """mH(Tensor(a) self) -> Tensor(a)"""

    # Taking the conjugate transpose of a real matrix is the same as the transpose
    return op.Einsum(self, equation="...ij->...ji")


@torch_op("aten::mH", complex=True, trace_only=True)
def aten_mH_complex(self: TFloat) -> TFloat:
    """mH(Tensor(a) self) -> Tensor(a)"""

    # TODO(#834): Allow calling scripted functions from other
    # scripted functions and remove trace only.

    # c is the last dimension being the real and imaginary parts
    trasposed = op.Einsum(self, equation="...ijc->...jic")
    return _complex_conjugate(trasposed)


@torch_op("aten::mT")
def aten_mT(self: TRealOrUInt8) -> TRealOrUInt8:
    """mT(Tensor(a) self) -> Tensor(a)"""

    return op.Einsum(self, equation="...ij->...ji")


@torch_op("aten::mT", complex=True)
def aten_mT_complex(self: TFloat) -> TFloat:
    """mT(Tensor(a) self) -> Tensor(a)"""

    # c is the last dimension being the real and imaginary parts
    return op.Einsum(self, equation="...ijc->...jic")


def aten_margin_ranking_loss(
    input1: TensorType,
    input2: TensorType,
    target: TensorType,
    margin: float = 0.0,
    reduction: int = 1,
) -> TensorType:
    """margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::masked_fill", "aten::masked_fill.Scalar", "aten::masked_fill.Tensor"))
def aten_masked_fill(self: TTensor, mask: BOOL, value: TTensor) -> TTensor:
    """masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor"""
    # NOTE: Do not attempt to cast `mask` to BOOL because mask should not take any other types.
    # `mask` coming in as other types is often an error and should fail the model.
    value_cast = op.CastLike(value, self)
    return op.Where(mask, value_cast, self)


def aten_masked_scatter(self: TensorType, mask: TensorType, source: TensorType) -> TensorType:
    """masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor"""

    raise NotImplementedError()


def aten_masked_select(self: TensorType, mask: TensorType) -> TensorType:
    """masked_select(Tensor self, Tensor mask) -> Tensor"""

    raise NotImplementedError()


def aten_masked_select_backward(
    grad: TensorType, input: TensorType, mask: TensorType
) -> TensorType:
    """masked_select_backward(Tensor grad, Tensor input, Tensor mask) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::matmul")
def aten_matmul(
    self: TRealUnlessInt16OrInt8, other: TRealUnlessInt16OrInt8
) -> TRealUnlessInt16OrInt8:
    """matmul(Tensor self, Tensor other) -> Tensor"""

    return op.MatMul(self, other)


def aten_matmul_backward(
    grad: TensorType, self: TensorType, other: TensorType, mask: Sequence[bool]
) -> tuple[TensorType, TensorType]:
    """matmul_backward(Tensor grad, Tensor self, Tensor other, bool[2] mask) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_matrix_H(self: TensorType) -> TensorType:
    """matrix_H(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


def aten_matrix_exp(self: TensorType) -> TensorType:
    """matrix_exp(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_matrix_exp_backward(self: TensorType, grad: TensorType) -> TensorType:
    """matrix_exp_backward(Tensor self, Tensor grad) -> Tensor"""

    raise NotImplementedError()


def aten_matrix_power(self: TensorType, n: int) -> TensorType:
    """matrix_power(Tensor self, int n) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::max")
def aten_max(self: TReal) -> TReal:
    """max(Tensor self) -> Tensor"""

    self_rank = op.Size(op.Shape(self))
    if self_rank == 0:
        self = op.Reshape(self, op.Constant(value_ints=[-1]))

    result = op.ReduceMax(self, keepdims=False)

    if self_rank == 0:
        result = op.Squeeze(result)

    return result


@torch_op("aten::max.dim")
def aten_max_dim(self: TReal, dim: int, keepdim: bool = False) -> Tuple[TReal, INT64]:
    """max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"""

    if op.Size(op.Shape(self)) == 0:
        result = self
        indices = op.Constant(value_int=0)
    else:
        dims = op.Reshape(dim, op.Constant(value_ints=[-1]))
        result = op.ReduceMax(self, dims, keepdims=keepdim)
        indices = op.ArgMax(self, axis=dim, keepdims=keepdim)
    return result, indices


@torch_op(("aten::maximum", "aten::max.other"))
def aten_maximum(self: TReal, other: TReal) -> TReal:
    """maximum(Tensor self, Tensor other) -> Tensor"""

    return op.Max(self, other)


@torch_op(("aten::maximum", "aten::max.other"))
def aten_maximum_bool(self: BOOL, other: BOOL) -> BOOL:
    """maximum(Tensor self, Tensor other) -> Tensor"""

    return op.Or(self, other)


@torch_op("aten::mean")
def aten_mean(self: TReal) -> TReal:
    """mean(Tensor self, *, ScalarType? dtype=None) -> Tensor"""

    result = op.ReduceMean(self)
    return op.Squeeze(result)


@torch_op("aten::mean.dim")
def aten_mean_dim(self: TReal, dim: INT64, keepdim: bool = False) -> TReal:
    """mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""

    if op.Size(op.Shape(self)) == 0:
        result = self
    else:
        if op.Size(op.Shape(dim)) == 0:
            dim = op.Unsqueeze(dim, axes=0)
        result = op.ReduceMean(self, axes=dim, keepdims=keepdim)
    return result


def aten_median(self: TensorType) -> TensorType:
    """median(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_meshgrid(tensors: Sequence[TensorType]) -> TensorType:
    """meshgrid(Tensor[] tensors) -> Tensor[]"""

    raise NotImplementedError()


@torch_op("aten::min")
def aten_min(self: TReal) -> TReal:
    """min(Tensor self) -> Tensor"""

    return op.ReduceMin(self, keepdims=False)


@torch_op("aten::min.dim")
def aten_min_dim(self: TReal, dim: int, keepdim: bool = False) -> Tuple[TReal, TInt]:
    """min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"""
    if op.Size(op.Shape(self)) == 0:
        result = self
        indices = op.Constant(value_int=0)
    else:
        dims = op.Reshape(dim, op.Constant(value_ints=[-1]))
        result = op.ReduceMin(self, dims, keepdims=keepdim)
        indices = op.ArgMin(self, axis=dim, keepdims=keepdim)

    return result, indices


@torch_op(("aten::minimum", "aten::min.other"))
def aten_minimum(self: TReal, other: TReal) -> TReal:
    """minimum(Tensor self, Tensor other) -> Tensor"""

    return op.Min(self, other)


@torch_op(("aten::minimum", "aten::min.other"))
def aten_minimum_bool(self: BOOL, other: BOOL) -> BOOL:
    """minimum(Tensor self, Tensor other) -> Tensor"""

    return op.And(self, other)


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
    """miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)"""

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
    """miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)"""

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
    """miopen_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor"""

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
    """miopen_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor"""

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
    """miopen_convolution_relu(Tensor self, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor"""

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
    """miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor"""

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
    """miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor"""

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
    """miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"""

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
    """miopen_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])"""

    raise NotImplementedError()


def aten_mkldnn_adaptive_avg_pool2d(
    self: TensorType, output_size: Sequence[int]
) -> TensorType:
    """mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor"""

    raise NotImplementedError()


def aten_mkldnn_adaptive_avg_pool2d_backward(
    grad_output: TensorType, self: TensorType
) -> TensorType:
    """mkldnn_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor"""

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
    """mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, int[] stride, int[] dilation, int groups) -> Tensor"""

    raise NotImplementedError()


def aten_mkldnn_linear_backward(
    self: TensorType, grad_output: TensorType, weight: TensorType, output_mask: Sequence[bool]
) -> tuple[TensorType, TensorType, TensorType]:
    """mkldnn_linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""

    raise NotImplementedError()


def aten_mkldnn_linear_backward_input(
    input_size: Sequence[int], grad_output: TensorType, weight: TensorType
) -> TensorType:
    """mkldnn_linear_backward_input(int[] input_size, Tensor grad_output, Tensor weight) -> Tensor"""

    raise NotImplementedError()


def aten_mkldnn_linear_backward_weights(
    grad_output: TensorType, input: TensorType, weight: TensorType, bias_defined: bool
) -> tuple[TensorType, TensorType]:
    """mkldnn_linear_backward_weights(Tensor grad_output, Tensor input, Tensor weight, bool bias_defined) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_mkldnn_max_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> TensorType:
    """mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor"""

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
    """mkldnn_max_pool2d_backward(Tensor grad_output, Tensor output, Tensor input, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor"""

    raise NotImplementedError()


def aten_mkldnn_max_pool3d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    ceil_mode: bool = False,
) -> TensorType:
    """mkldnn_max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor"""

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
    """mkldnn_max_pool3d_backward(Tensor grad_output, Tensor output, Tensor input, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::mm")
def aten_mm(
    self: TRealUnlessInt16OrInt8, mat2: TRealUnlessInt16OrInt8
) -> TRealUnlessInt16OrInt8:
    """mm(Tensor self, Tensor mat2) -> Tensor"""

    return op.MatMul(self, mat2)


def aten_mode(
    self: TensorType, dim: int = -1, keepdim: bool = False
) -> tuple[TensorType, TensorType]:
    """mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)"""

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
    """mps_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""

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
    """mps_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool[2] output_mask) -> (Tensor, Tensor)"""

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
    """mps_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor"""

    raise NotImplementedError()


def aten_msort(self: TensorType) -> TensorType:
    """msort(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::mul", "aten::mul.Tensor"))
def aten_mul(self: TReal, other: TReal) -> TReal:
    """mul.Tensor(Tensor self, Tensor other) -> Tensor"""
    # FIXME(titaiwang): get rid of this when we have type_promotion
    other = op.CastLike(other, self)
    return op.Mul(self, other)


@torch_op(("aten::mul", "aten::mul.Tensor"))
def aten_mul_bool(self: BOOL, other: BOOL) -> BOOL:
    """ONNX Mul doesn't support Boolean, so use And as an equivalent operator."""

    # TODO(justinchuby): Handle cases where type reconcilation is not enough,
    # since different ONNX operators are used based on different data types.

    return op.And(self, other)


@torch_op("aten::multinomial")
def aten_multinomial(
    self: TFloat,
    num_samples: int,
    replacement: bool = False,  # pylint: disable=unused-argument
) -> TInt:
    """multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor"""
    # ONNX Multinomial doesn't support 1D input
    if op.Size(op.Shape(self)) == 1:
        unsqueezed_input = op.Unsqueeze(self, axes=0)
    else:
        unsqueezed_input = self
    # ONNX multinomial expects log probability
    log_input = op.Log(unsqueezed_input)
    result = op.Multinomial(log_input, dtype=INT64.dtype, sample_size=num_samples)
    if op.Size(op.Shape(self)) == 1:
        result = op.Squeeze(result)
    return result


def aten_multiply(self: TensorType, other: TensorType) -> TensorType:
    """multiply.Tensor(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_mv(self: TensorType, vec: TensorType) -> TensorType:
    """mv(Tensor self, Tensor vec) -> Tensor"""

    raise NotImplementedError()


def aten_mvlgamma(self: TensorType, p: int) -> TensorType:
    """mvlgamma(Tensor self, int p) -> Tensor"""

    raise NotImplementedError()


def aten_nan_to_num(
    self: TensorType,
    nan: Optional[float] = None,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
) -> TensorType:
    """nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor"""

    raise NotImplementedError()


def aten_nanmean(
    self: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    dtype: Optional[int] = None,
) -> TensorType:
    """nanmean(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""

    raise NotImplementedError()


def aten_nanmedian(self: TensorType) -> TensorType:
    """nanmedian(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_nanquantile(
    self: TensorType,
    q: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    interpolation: str = "linear",
) -> TensorType:
    """nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor"""

    raise NotImplementedError()


def aten_nansum(
    self: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    dtype: Optional[int] = None,
) -> TensorType:
    """nansum(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::narrow")
def aten_narrow(self: TTensor, dim: INT64, start: INT64, length: INT64) -> TTensor:
    """narrow(Tensor(a) self, int dim, SymInt start, SymInt length) -> Tensor(a)"""

    dim_rank = op.Size(op.Shape(dim))
    if dim_rank == 0:
        dim = op.Reshape(dim, op.Constant(value_ints=[-1]))

    start_rank = op.Size(op.Shape(start))
    if start_rank == 0:
        start = op.Reshape(start, op.Constant(value_ints=[-1]))

    length_rank = op.Size(op.Shape(length))
    if length_rank == 0:
        length = op.Reshape(length, op.Constant(value_ints=[-1]))

    end = op.Add(start, length)
    result = op.Slice(self, start, end, dim)
    return result


def aten_narrow_copy(self: TensorType, dim: int, start: INT64, length: INT64) -> TensorType:
    """narrow_copy(Tensor self, int dim, SymInt start, SymInt length) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::native_batch_norm", trace_only=True)
def aten_native_batch_norm(
    input: TFloat,
    weight: Optional[TFloat] = None,
    bias: Optional[TFloat] = None,
    running_mean: Optional[TFloat] = None,
    running_var: Optional[TFloat] = None,
    training: bool = False,
    momentum: float = 0.9,
    eps: float = 1e-05,
) -> Tuple[TFloat, TFloat, TFloat]:
    """native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)"""

    if weight is None:  # Set to 1.0 as default
        weight = op.Expand(op.Constant(value_floats=[1.0]), op.Shape(input, start=1, end=2))

    if bias is None:  # Set to 0.0 as default
        bias = op.Expand(op.Constant(value_floats=[0.0]), op.Shape(input, start=1, end=2))

    axes = list(range(len(input.shape)))
    axes.pop(1)
    axes = op.Constant(value_ints=axes)
    if running_mean is None:  # Using input mean
        running_mean = op.Squeeze(op.ReduceMean(input, axes))

    if running_var is None:  # Using input var
        mean = op.ReduceMean(input, axes)
        input_sub_mean = op.Sub(input, mean)
        sqr_input_sub_mean = op.Mul(input_sub_mean, input_sub_mean)
        running_var = op.Squeeze(op.ReduceMean(sqr_input_sub_mean, axes))

    # Have to split to 2 private functions, because training_function return 3 outputs
    # While inference_function return 1 output
    if training is True:
        norm, mean, var = _aten_native_batch_norm_training_onnx(
            input, weight, bias, running_mean, running_var, axes, training, momentum, eps
        )
    else:
        norm, mean, var = _aten_native_batch_norm_inference_onnx(
            input, weight, bias, running_mean, running_var, training, momentum, eps
        )
    return norm, mean, var


@torch_op("aten::native_batch_norm", private=True)
def _aten_native_batch_norm_training_onnx(
    input: TFloat,
    weight: TFloat,
    bias: TFloat,
    running_mean: TFloat,
    running_var: TFloat,
    axes: INT64,
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[TFloat, TFloat, TFloat]:
    # Assert(training is True)
    norm, running_mean, running_var = op.BatchNormalization(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon=eps,
        momentum=momentum,
        training_mode=training,
    )
    # Compute var and rstd
    mean = op.ReduceMean(input, axes)
    input_sub_mean = op.Sub(input, mean)
    sqr = op.Mul(input_sub_mean, input_sub_mean)
    var = op.ReduceMean(sqr, axes, keepdims=False)
    rstd = op.Div(1.0, op.Sqrt(var + eps))
    # Get mean again with size = [1, C]
    mean = op.ReduceMean(input, axes, keepdims=False)
    return norm, mean, rstd


@torch_op("aten::native_batch_norm", private=True)
def _aten_native_batch_norm_inference_onnx(
    input: TFloat,
    weight: TFloat,
    bias: TFloat,
    running_mean: TFloat,
    running_var: TFloat,
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[TFloat, TFloat, TFloat]:
    # Assert(training is False)
    norm = op.BatchNormalization(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon=eps,
        momentum=momentum,
        training_mode=training,
    )
    # Cannot return 2 dup output, so have to do twice with different variable name
    empty_mean = op.Cast(op.Shape(input, start=0, end=0), to=FLOAT.dtype)
    empty_var = op.Cast(op.Shape(input, start=0, end=0), to=FLOAT.dtype)
    return norm, empty_mean, empty_var


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
    """native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""

    raise NotImplementedError()


def aten_native_channel_shuffle(self: TensorType, groups: int) -> TensorType:
    """native_channel_shuffle(Tensor self, int groups) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::native_dropout")
def aten_native_dropout(
    input: TFloatOrBFloat16, p: float, train: bool = True
) -> Tuple[TFloatOrBFloat16, BOOL]:
    """native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)"""

    # Python bool attributes need to be explicitly converted to BOOL
    # because the underlying attribute type is int
    # TODO(#872): Allow ONNX Script to handle this conversion
    train = op.Cast(train, to=BOOL.dtype)
    result, mask = op.Dropout(input, p, train)
    return result, mask


def aten_native_dropout_backward(
    grad_output: TensorType, mask: TensorType, scale: float
) -> TensorType:
    """native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::native_group_norm", trace_only=True)
def aten_native_group_norm(
    input: TFloat,
    weight: Optional[TFloat] = None,
    bias: Optional[TFloat] = None,
    N: Optional[INT64] = None,  # pylint: disable=unused-argument
    C: Optional[INT64] = None,  # pylint: disable=unused-argument
    HxW: Optional[INT64] = None,  # pylint: disable=unused-argument
    group: int = 1,
    eps: float = 1e-05,
) -> Tuple[TFloat, TFloat, TFloat]:
    """native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)"""

    # Actually we don't need N,C,HxW value because the input tensor has that information
    if weight is None:  # Set to 1.0 as default, the shape is Channel size
        weight = op.Expand(op.Constant(value_floats=[1.0]), op.Shape(input, start=1, end=2))

    if bias is None:  # Set to 0.0 as default, the shape is Channel size
        bias = op.Expand(op.Constant(value_floats=[0.0]), op.Shape(input, start=1, end=2))

    # Accoding to Torch, return rstd instead of var
    norm, mean, rstd = _aten_native_group_norm_onnx(input, weight, bias, group, eps)
    return norm, mean, rstd


@torch_op("aten::native_group_norm", private=True)
def _aten_native_group_norm_onnx(
    input: TFloat,
    weight: TFloat,
    bias: TFloat,
    group: INT64,
    eps: float,
) -> Tuple[TFloat, TFloat, TFloat]:
    # Because onnx.GroupNorm() need size=group for weight and bias
    # But the torch's aten function's input need size=channel, the size mismatched
    # So we have to use onnx.InstanceNorm() to simulate
    neg_1 = op.Constant(value_ints=[-1])
    # Create weight_instance_norm and bias_instance_norm, copied from Torch ONNX converter
    group_tensor = op.Reshape(group, neg_1)
    # 0 in the shape list keeps dimension value unchanged, for InstanceNorm need [0,group,-1]
    shape_input = op.Concat(op.Constant(value_ints=[0]), group_tensor, neg_1, axis=0)
    input_reshaped = op.Reshape(input, shape_input)
    weight_inst_norm = op.Expand(op.Constant(value_floats=[1.0]), group_tensor)
    bias_inst_norm = op.Expand(op.Constant(value_floats=[0.0]), group_tensor)
    norm = op.InstanceNormalization(
        input_reshaped, weight_inst_norm, bias_inst_norm, epsilon=eps
    )
    # Reshape back to input's shape
    norm = op.Reshape(norm, op.Shape(input))
    # Using the input weight and bias to do affine
    # But need to unsqueeze to the target shape for broading cast easy
    input_rank = op.Size(op.Shape(input))
    axes_unsqueeze = op.Range(1, input_rank - 1, 1)
    weight_full_shape = op.Unsqueeze(weight, axes_unsqueeze)
    bias_full_shape = op.Unsqueeze(bias, axes_unsqueeze)
    norm_mul_weight = op.Mul(norm, weight_full_shape)
    norm_result = op.Add(norm_mul_weight, bias_full_shape)
    # Compute mean and rstd, but using Torch algorithm
    # The returned shape for mean and vstd should be [N, group, -1]
    N = op.Shape(input, start=0, end=1)
    shape_N_group_neg1 = op.Concat(N, group_tensor, neg_1, axis=0)
    input_N_group_neg1 = op.Reshape(input, shape_N_group_neg1)
    # The output size is [N, group], so dims = [2]
    axes = op.Constant(value_ints=[2])
    # Get mean which size is [N, group, 1], for broadcasting
    mean = op.ReduceMean(input_N_group_neg1, axes)
    input_sub_mean = op.Sub(input_N_group_neg1, mean)
    sqr_input_sub_mean = op.Mul(input_sub_mean, input_sub_mean)
    # In Pytorch, vstd = 1/(sqrt(var + eps))
    var = op.ReduceMean(sqr_input_sub_mean, axes, keepdims=False)
    rstd = op.Div(1.0, op.Sqrt(var + eps))
    # Get the correct shape [N, group] for mean again
    mean = op.ReduceMean(input_N_group_neg1, axes, keepdims=False)
    return norm_result, mean, rstd


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
    """native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""

    raise NotImplementedError()


@torch_op("aten::native_layer_norm", trace_only=True)
def aten_native_layer_norm(
    input: TReal,
    normalized_shape: INT64,
    weight: Optional[TReal] = None,
    bias: Optional[TReal] = None,
    eps: float = 1e-05,
) -> Tuple[TReal, TReal, TReal]:
    """native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)"""

    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
    # The mean and standard-deviation are calculated over the last D dimensions,
    # where D is the dimension of normalized_shape. For example, if normalized_shape is
    # (3, 5) (a 2-dimensional shape), the mean and standard-deviation are computed
    # over the last 2 dimensions of the input (i.e. input.mean((-2, -1))).

    # Use Python to manipulate
    start_axis = -len(normalized_shape)

    if weight is None:
        one = op.Constant(value_floats=[1.0])
        weight = op.Expand(one, op.Shape(input, start=start_axis))
        weight = op.CastLike(weight, input)

    result, mean, rdenominator = op.LayerNormalization(
        input, weight, bias, axis=start_axis, epsilon=eps
    )

    return result, mean, rdenominator


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
    """native_layer_norm_backward(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"""

    raise NotImplementedError()


def aten_native_norm(self: TensorType, p: float = 2.0) -> TensorType:
    """native_norm(Tensor self, Scalar p=2) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::ne", "aten::ne.Scalar", "aten::ne.Tensor"))
def aten_ne(self: TReal, other: TReal) -> BOOL:
    """ne.Tensor(Tensor self, Tensor other) -> Tensor"""

    return op.Not(op.Equal(self, other))


@torch_op("aten::neg")
def aten_neg(self: TReal) -> TReal:
    """neg(Tensor self) -> Tensor"""

    return op.Neg(self)


def aten_negative(self: TensorType) -> TensorType:
    """negative(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::new_empty")
def aten_new_empty(self: TTensor, size: INT64) -> TTensor:
    """new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    # using zero to simulate empty array
    result = op.ConstantOfShape(size)
    return op.CastLike(result, self)


@torch_op("aten::new_empty")
def aten_new_empty_dtype(
    self: TTensor,  # pylint: disable=unused-argument
    size: INT64,
    dtype: int,
) -> TTensor:
    """new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    # using zero to simulate empty array
    result = op.ConstantOfShape(size)
    return op.Cast(result, to=dtype)


@torch_op("aten::new_empty_strided")
def aten_new_empty_strided(
    self: TTensor,
    size: INT64,
    stride: INT64,  # pylint: disable=unused-argument
) -> TTensor:
    """new_empty_strided(Tensor self, SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    # using zero to simulate empty array
    zero = op.ConstantOfShape(size)
    return op.CastLike(zero, self)


@torch_op("aten::new_empty_strided")
def aten_new_empty_strided_dtype(
    self: TTensor,  # pylint: disable=unused-argument
    size: INT64,
    stride: INT64,  # pylint: disable=unused-argument
    dtype: int,
) -> TTensor:
    """new_empty_strided(Tensor self, SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    # using zero to simulate empty array
    zero = op.ConstantOfShape(size)
    return op.Cast(zero, to=dtype)


@torch_op("aten::new_full")
def aten_new_full(self: TTensor, size: INT64, fill_value: TTensor) -> TTensor:
    # new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    fill_value = op.CastLike(fill_value, self)
    return op.Expand(fill_value, size)


@torch_op("aten::new_full")
def aten_new_full_dtype(
    self: TTensor,  # pylint: disable=unused-argument
    size: INT64,
    fill_value: TTensor,
    dtype: int,
) -> TTensor:
    # new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    fill_value = op.Cast(fill_value, to=dtype)
    return op.Expand(fill_value, size)


@torch_op("aten::new_ones")
def aten_new_ones(self: TReal, size: INT64) -> TReal:  # pylint: disable=unused-argument
    """new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    one = op.Constant(value_float=1.0)
    result = op.Expand(one, size)
    return op.CastLike(result, self)


@torch_op("aten::new_ones")
def aten_new_ones_dtype(
    self: TReal,  # pylint: disable=unused-argument
    size: INT64,
    dtype: int,
) -> TReal:
    one = op.Constant(value_float=1.0)
    result = op.Expand(one, size)
    return op.Cast(result, to=dtype)


@torch_op("aten::new_zeros")
def aten_new_zeros(self: TReal, size: INT64) -> TReal:
    """new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    result = op.ConstantOfShape(size)
    return op.CastLike(result, self)


@torch_op("aten::new_zeros")
def aten_new_zeros_dtype(
    self: TReal,  # pylint: disable=unused-argument
    size: INT64,
    dtype: int,
) -> TReal:
    result = op.ConstantOfShape(size)
    return op.Cast(result, to=dtype)


def aten_nextafter(self: TensorType, other: TensorType) -> TensorType:
    """nextafter(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::nonzero")
def aten_nonzero(self: TTensor) -> INT64:
    """nonzero(Tensor self) -> Tensor"""
    # NOTE: In torch the return shape is [n, d], while in onnx [d, n],
    # where `d` is rank of input tensor, `n` is number of nonzero elements.
    return op.Transpose(op.NonZero(self), perm=[1, 0])


def aten_nonzero_numpy(self: TensorType) -> TensorType:
    """nonzero_numpy(Tensor self) -> Tensor[]"""

    raise NotImplementedError()


def aten_norm_except_dim(v: TensorType, pow: int = 2, dim: int = 0) -> TensorType:
    """norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::normal", "aten::normal_functional"))
def aten_normal(
    self: TTensor,
    mean: float = 0.0,
    std: float = 1.0,
) -> TFloat:  # type: ignore[type-var]
    """normal_functional(Tensor self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor"""

    self_rank = op.Size(op.Shape(self))
    if self_rank == 0:
        self = op.Reshape(self, op.Constant(value_ints=[-1]))

    result = op.RandomNormalLike(self, mean=mean, scale=std)
    return result


@torch_op("aten::normal.float_float")
def aten_normal_float_float(
    mean: float, std: float, size: INT64, dtype: int = FLOAT.dtype
) -> TensorType:
    """normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    # Create a dummy tensor for RandomNormalLike to get the shape
    dummy_tensor = op.ConstantOfShape(size)
    result = op.RandomNormalLike(dummy_tensor, mean=mean, scale=std)
    return op.Cast(result, to=dtype)


@torch_op("aten::normal.float_Tensor")
def aten_normal_float_tensor(mean: FLOAT, std: TFloat) -> TFloat:
    """normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor"""

    mean_casted = op.CastLike(mean, std)
    sampled = op.RandomNormalLike(mean_casted, mean=0.0, scale=1.0)
    # Transform the distribution to the mean and std
    return op.Add(op.Mul(std, sampled), mean_casted)


@torch_op("aten::normal.Tensor_float")
def aten_normal_tensor_float(mean: TFloat, std: FLOAT) -> TFloat:
    """normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor"""

    sampled = op.RandomNormalLike(mean, mean=0.0, scale=1.0)
    # Transform the distribution to the mean and std
    return op.Add(op.Mul(op.CastLike(std, sampled), sampled), mean)


@torch_op("aten::normal.Tensor_Tensor")
def aten_normal_tensor_tensor(mean: TFloat, std: TFloat) -> TFloat:
    """normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor"""

    sampled = op.RandomNormalLike(mean, mean=0.0, scale=1.0)
    # Transform the distribution to the mean and std
    return op.Add(op.Mul(std, sampled), mean)


def aten_not_equal(self: TensorType, other: TensorType) -> TensorType:
    """not_equal.Tensor(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_nuclear_norm(self: TensorType, keepdim: bool = False) -> TensorType:
    """nuclear_norm(Tensor self, bool keepdim=False) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::ones")
def aten_ones(size: IntType, dtype: int = FLOAT.dtype):
    """ones(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    size = op.Cast(size, to=INT64.dtype)
    one = op.Constant(value_float=1.0)
    one = op.Cast(one, to=dtype)
    return op.Expand(one, size)


@torch_op("aten::ones_like", trace_only=True)
def aten_ones_like(self: TTensor, dtype: int = -1) -> TTensor:
    """ones_like.

    Note: dtype is an onnx enum. Users should convert torch dtype to onnx dtype
    before calling this function.
    """
    # ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    # NOTE: trace_only because both if branches need to be the same type, but we have
    # a cast in the if branch.

    if dtype == -1:
        one = op.CastLike(1, self)
    else:
        one = op.Cast(1, to=dtype)
    return _aten_ones_like_onnx(self, one)


@torch_op("aten::ones_like", private=True)
def _aten_ones_like_onnx(self: TTensor, one) -> TTensor:
    shape = op.Shape(self)
    return op.Expand(one, shape)


def aten_or(self: TensorType, other: TensorType) -> TensorType:
    """__or__.Tensor(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_orgqr(self: TensorType, input2: TensorType) -> TensorType:
    """orgqr(Tensor self, Tensor input2) -> Tensor"""

    raise NotImplementedError()


def aten_ormqr(
    self: TensorType,
    input2: TensorType,
    input3: TensorType,
    left: bool = True,
    transpose: bool = False,
) -> TensorType:
    """ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor"""

    raise NotImplementedError()


def aten_outer(self: TensorType, vec2: TensorType) -> TensorType:
    """outer(Tensor self, Tensor vec2) -> Tensor"""

    raise NotImplementedError()


def aten_output_nr(self: TensorType) -> int:
    """output_nr(Tensor self) -> int"""

    raise NotImplementedError()


def aten_pairwise_distance(
    x1: TensorType, x2: TensorType, p: float = 2.0, eps: float = 1e-06, keepdim: bool = False
) -> TensorType:
    """pairwise_distance(Tensor x1, Tensor x2, float p=2, float eps=1e-06, bool keepdim=False) -> Tensor"""

    raise NotImplementedError()


def aten_pdist(self: TensorType, p: float = 2.0) -> TensorType:
    """pdist(Tensor self, float p=2) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::permute", trace_only=True)
def aten_permute(self: TTensor, dims: Sequence[int]) -> TTensor:
    """permute(Tensor(a) self, int[] dims) -> Tensor(a)"""

    if not dims:
        return op.Transpose(self)

    # Handle negative axes
    dims = [axis + len(dims) if axis < 0 else axis for axis in dims]

    return op.Transpose(self, perm=dims)


def aten_permute_copy(self: TensorType, dims: Sequence[int]) -> TensorType:
    """permute_copy(Tensor self, int[] dims) -> Tensor"""

    raise NotImplementedError()


def aten_pin_memory(self: TensorType, device: Optional[str] = None) -> TensorType:
    """pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)"""

    raise NotImplementedError()


def aten_pinverse(self: TensorType, rcond: float = 1e-15) -> TensorType:
    """pinverse(Tensor self, float rcond=1e-15) -> Tensor"""

    raise NotImplementedError()


def aten_pixel_shuffle(self: TensorType, upscale_factor: int) -> TensorType:
    """pixel_shuffle(Tensor self, int upscale_factor) -> Tensor"""

    raise NotImplementedError()


def aten_pixel_unshuffle(self: TensorType, downscale_factor: int) -> TensorType:
    """pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor"""

    raise NotImplementedError()


def aten_poisson(self: TensorType, generator: Optional[str] = None) -> TensorType:
    """poisson(Tensor self, Generator? generator=None) -> Tensor"""

    raise NotImplementedError()


def aten_poisson_nll_loss(
    input: TensorType,
    target: TensorType,
    log_input: bool,
    full: bool,
    eps: float,
    reduction: int,
) -> TensorType:
    """poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor"""

    raise NotImplementedError()


def aten_polar(abs: TensorType, angle: TensorType) -> TensorType:
    """polar(Tensor abs, Tensor angle) -> Tensor"""

    raise NotImplementedError()


def aten_polygamma(n: int, self: TensorType) -> TensorType:
    """polygamma(int n, Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_positive(self: TensorType) -> TensorType:
    """positive(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


@torch_op(("aten::pow", "aten::pow.Tensor_Tensor", "aten::pow.Tensor_Scalar"))
def aten_pow(self: TReal, exponent: TTensor) -> TReal:
    """pow(Tensor self, Tensor exponent) -> Tensor"""

    return op.Pow(self, exponent)


def aten_prelu(self: TensorType, weight: TensorType) -> TensorType:
    """prelu(Tensor self, Tensor weight) -> Tensor"""

    raise NotImplementedError()


def aten_prelu_backward(
    grad_output: TensorType, self: TensorType, weight: TensorType
) -> tuple[TensorType, TensorType]:
    """prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_prod(self: TensorType, dtype: Optional[int] = None) -> TensorType:
    """prod(Tensor self, *, ScalarType? dtype=None) -> Tensor"""

    raise NotImplementedError()


def aten_promote_types(type1: int, type2: int) -> int:
    """promote_types(ScalarType type1, ScalarType type2) -> ScalarType"""

    raise NotImplementedError()


def aten_put(
    self: TensorType, index: TensorType, source: TensorType, accumulate: bool = False
) -> TensorType:
    """put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor"""

    raise NotImplementedError()


def aten_q_per_channel_axis(self: TensorType) -> int:
    """q_per_channel_axis(Tensor self) -> int"""

    raise NotImplementedError()


def aten_q_per_channel_scales(self: TensorType) -> TensorType:
    """q_per_channel_scales(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_q_per_channel_zero_points(self: TensorType) -> TensorType:
    """q_per_channel_zero_points(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_q_scale(self: TensorType) -> float:
    """q_scale(Tensor self) -> float"""

    raise NotImplementedError()


def aten_q_zero_point(self: TensorType) -> int:
    """q_zero_point(Tensor self) -> int"""

    raise NotImplementedError()


def aten_qr(self: TensorType, some: bool = True) -> tuple[TensorType, TensorType]:
    """qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)"""

    raise NotImplementedError()


def aten_qscheme(self: TensorType) -> str:
    """qscheme(Tensor self) -> QScheme"""

    raise NotImplementedError()


def aten_quantile(
    self: TensorType,
    q: TensorType,
    dim: Optional[int] = None,
    keepdim: bool = False,
    interpolation: str = "linear",
) -> TensorType:
    """quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor"""

    raise NotImplementedError()


def aten_quantize_per_channel(
    self: TensorType, scales: TensorType, zero_points: TensorType, axis: int, dtype: int
) -> TensorType:
    """quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) -> Tensor"""

    raise NotImplementedError()


def aten_quantize_per_tensor(
    self: TensorType, scale: float, zero_point: int, dtype: int
) -> TensorType:
    """quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor"""

    raise NotImplementedError()


def aten_quantize_per_tensor_dynamic(
    self: TensorType, dtype: int, reduce_range: bool
) -> TensorType:
    """quantize_per_tensor_dynamic(Tensor self, ScalarType dtype, bool reduce_range) -> Tensor"""

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
    """quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor"""

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
    """quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor"""

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
    """quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)"""

    raise NotImplementedError()


def aten_quantized_max_pool1d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    ceil_mode: bool = False,
) -> TensorType:
    """quantized_max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor"""

    raise NotImplementedError()


def aten_quantized_max_pool2d(
    self: TensorType,
    kernel_size: Sequence[int],
    stride: Optional[Sequence[int]] = None,
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> TensorType:
    """quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor"""

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
    """quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor"""

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
    """quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor"""

    raise NotImplementedError()


def aten_rad2deg(self: TensorType) -> TensorType:
    """rad2deg(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::rand")
def aten_rand(size: INT64, dtype: int = FLOAT.dtype) -> TReal:
    """rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    shaper = op.ConstantOfShape(size)
    return op.RandomUniformLike(shaper, dtype=dtype)


@torch_op("aten::rand_like")
def aten_rand_like(self: TFloat) -> TFloat:
    """rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    return op.RandomUniformLike(self)


@torch_op("aten::rand_like")
def aten_rand_like_dtype(self: TensorType, dtype: int) -> TensorType:
    """rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    return op.RandomUniformLike(self, dtype=dtype)


@torch_op("aten::randint")
def aten_randint(high: INT64, size: INT64, dtype: int = INT64.dtype) -> TensorType:
    """randint(SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    shaper = op.ConstantOfShape(size)
    rand = op.RandomUniformLike(shaper)
    # Scale to [0, high] first
    rand_scaled = op.Mul(rand, op.CastLike(high, rand))
    # Round to ints
    rand_int = op.Floor(rand_scaled)
    return op.Cast(rand_int, to=dtype)


@torch_op("aten::randint.low")
def aten_randint_low(
    low: INT64, high: INT64, size: INT64, dtype: int = INT64.dtype
) -> TensorType:
    """randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    shaper = op.ConstantOfShape(size)
    rand = op.RandomUniformLike(shaper)
    # Translate to [low, high] first
    high = op.Cast(high, to=FLOAT.dtype)
    low = op.Cast(low, to=FLOAT.dtype)
    rand_translated = op.Add(op.Mul(rand, op.Sub(high, low)), low)
    # Round to ints
    rand_int = op.Floor(rand_translated)
    return op.Cast(rand_int, to=dtype)


@torch_op("aten::randint_like")
def aten_randint_like(self: TensorType, high: INT64) -> IntType:
    """randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    self_float = op.Cast(self, to=FLOAT.dtype)
    rand = op.RandomUniformLike(self_float)
    # Scale to [0, high] first
    rand_scaled = op.Mul(rand, op.CastLike(high, rand))
    # Round to ints
    rand_int = op.Floor(rand_scaled)
    return op.CastLike(rand_int, self)


@torch_op("aten::randint_like")
def aten_randint_like_dtype(self: TensorType, high: INT64, dtype: int) -> TensorType:
    """randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    self_float = op.Cast(self, to=FLOAT.dtype)
    rand = op.RandomUniformLike(self_float)
    # Scale to [0, high] first
    rand_scaled = op.Mul(rand, op.CastLike(high, rand))
    # Round to ints
    rand_int = op.Floor(rand_scaled)
    return op.Cast(rand_int, to=dtype)


@torch_op("aten::randint_like.low_dtype")
def aten_randint_like_low_dtype(self: TensorType, low: INT64, high: INT64) -> IntType:
    """randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    This is the TorchLib overload for aten::randint_like.low_dtype when dtype is None.
    """

    self_float = op.Cast(self, to=FLOAT.dtype)
    rand = op.RandomUniformLike(self_float)
    # Translate to [low, high] first
    high = op.Cast(high, to=FLOAT.dtype)
    low = op.Cast(low, to=FLOAT.dtype)
    rand_translated = op.Add(op.Mul(rand, op.Sub(high, low)), low)
    # Round to ints
    rand_int = op.Floor(rand_translated)
    return op.CastLike(rand_int, self)


@torch_op("aten::randint_like.low_dtype")
def aten_randint_like_low_dtype_dtype(
    self: TensorType, low: INT64, high: INT64, dtype: int
) -> TensorType:
    """randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    self_float = op.Cast(self, to=FLOAT.dtype)
    rand = op.RandomUniformLike(self_float)
    # Translate to [low, high] first
    high = op.Cast(high, to=FLOAT.dtype)
    low = op.Cast(low, to=FLOAT.dtype)
    rand_translated = op.Add(op.Mul(rand, op.Sub(high, low)), low)
    # Round to ints
    rand_int = op.Floor(rand_translated)
    return op.Cast(rand_int, to=dtype)


@torch_op("aten::randn")
def aten_randn(size: INT64, dtype: int = FLOAT.dtype) -> TReal:
    """randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    shaper = op.ConstantOfShape(size)
    return op.RandomNormalLike(shaper, dtype=dtype)


@torch_op("aten::randn_like")
def aten_randn_like(self: TFloat) -> TFloat:
    """randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    return op.RandomNormalLike(self)


@torch_op("aten::randn_like")
def aten_randn_like_dtype(self: TensorType, dtype: int) -> TensorType:
    """randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    return op.RandomNormalLike(self, dtype=dtype)


def aten_randperm(n: int) -> TensorType:
    """randperm(int n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


def aten_range(start: float, end: float) -> TensorType:
    """range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


def aten_ravel(self: TensorType) -> TensorType:
    """ravel(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


def aten_real(self: TensorType) -> TensorType:
    """real(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


@torch_op("aten::reciprocal")
def aten_reciprocal(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """reciprocal(Tensor self) -> Tensor"""

    return op.Reciprocal(self)


def aten_record_stream(self: TensorType, s: str) -> Any:
    """record_stream(Tensor(a!) self, Stream s) -> ()"""

    raise NotImplementedError()


def aten_refine_names(self: TensorType, names: Sequence[str]) -> TensorType:
    """refine_names(Tensor(a) self, Dimname[] names) -> Tensor(a)"""

    raise NotImplementedError()


@torch_op("aten::remainder")
def aten_remainder(self: TFloatOrBFloat16, other: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """remainder.Tensor(Tensor self, Tensor other) -> Tensor"""

    # TODO(justinchuby): Improve fp16 precision by following the logic in
    # https://github.com/pytorch/pytorch/blob/3a823e46170778cc32783f27596c77d0103084a9/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L264-L277

    # a - a.div(b, rounding_mode="floor") * b
    rounded_quotient = op.Floor(op.Div(self, other))

    return op.Sub(self, op.Mul(rounded_quotient, other))


@torch_op("aten::remainder")
def aten_remainder_int(self: TInt, other: TInt) -> TInt:
    """remainder.Tensor(Tensor self, Tensor other) -> Tensor"""

    return op.Mod(self, other)


def aten_rename(self: TensorType, names: Optional[str]) -> TensorType:
    """rename(Tensor(a) self, Dimname[]? names) -> Tensor(a)"""

    raise NotImplementedError()


def aten_renorm(self: TensorType, p: float, dim: int, maxnorm: float) -> TensorType:
    """renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::repeat")
def aten_repeat(self: TTensor, repeats: TInt) -> TTensor:
    """repeat(Tensor self, SymInt[] repeats) -> Tensor"""

    if op.Size(repeats) == 0:
        result = self
    else:
        # TODO(justinchuby): Make ones_like a function when onnxscript supports it
        repeats = op.Cast(repeats, to=INT64.dtype)
        # shape = ones_like(repeats) := {
        one = op.Constant(value_int=1)
        repeats_shape = op.Shape(repeats)
        shape = op.Expand(one, repeats_shape)
        # }
        self_expanded = op.Expand(self, shape)
        result = op.Tile(self_expanded, repeats)
    return result


def aten_repeat_interleave(
    repeats: TensorType, output_size: Optional[int] = None
) -> TensorType:
    """repeat_interleave.Tensor(Tensor repeats, *, int? output_size=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::reshape")
def aten_reshape(self: TTensor, shape: IntType) -> TTensor:
    """reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)"""

    # Reshape only support INT64 as 'shape'
    shape = op.Cast(shape, to=INT64.dtype)
    return op.Reshape(self, shape)


def aten_reshape_as(self: TensorType, other: TensorType) -> TensorType:
    """reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)"""

    raise NotImplementedError()


@torch_op("aten::resolve_conj")
def aten_resolve_conj(self: TTensor) -> TTensor:
    """resolve_conj(Tensor(a) self) -> Tensor(a)"""

    return op.Identity(self)


@torch_op("aten::resolve_neg")
def aten_resolve_neg(self: TTensor) -> TTensor:
    """resolve_neg(Tensor(a) self) -> Tensor(a)"""

    return op.Identity(self)


def aten_result_type(tensor: TensorType, other: TensorType) -> int:
    """result_type.Tensor(Tensor tensor, Tensor other) -> ScalarType"""

    raise NotImplementedError()


def aten_retain_grad(self: TensorType) -> Any:
    """retain_grad(Tensor(a!) self) -> ()"""

    raise NotImplementedError()


def aten_retains_grad(self: TensorType) -> bool:
    """retains_grad(Tensor self) -> bool"""

    raise NotImplementedError()


def aten_rnn_relu_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> TensorType:
    """rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor"""

    raise NotImplementedError()


def aten_rnn_tanh_cell(
    input: TensorType,
    hx: TensorType,
    w_ih: TensorType,
    w_hh: TensorType,
    b_ih: Optional[TensorType] = None,
    b_hh: Optional[TensorType] = None,
) -> TensorType:
    """rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::roll", trace_only=True)
def aten_roll(self: TTensor, shifts: INT64, dims: Sequence[int] = ()) -> TTensor:
    """roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor"""

    self_rank = len(self.shape)
    if self_rank == 0:
        return self
    elif self.shape[0] == 0:  # empty tensor
        return self
    else:
        if isinstance(dims, tuple) and len(dims) == 0:  # Empty list
            # assert isinstance(shifts, int)
            return _aten_roll_shift_no_dim_onnx(self, shifts)
        else:
            # assert len(shifts) == len(dims), but shifts is a tensor, dims is a list
            result = self
            for i in range(len(shifts)):  # pylint: disable=consider-using-enumerate
                shift = op.Gather(shifts, i, axis=0)
                dim = dims[i]
                result = _aten_roll_shift_and_dim_onnx(result, shift, dim)
            return result


@torch_op("aten::roll", private=True)
def _aten_roll_shift_no_dim_onnx(self: TTensor, shift: INT64) -> TTensor:
    neg_1 = op.Constant(value_ints=[-1])
    # flatten the self tensor: from [[A,B],[C,D]] to [A,B,C,D]
    self_flatten = op.Reshape(self, neg_1)
    # Compute slice length
    shift_tensor = op.Reshape(shift, neg_1)
    if shift_tensor < 0:
        # For [A,B,C,D], if shift is -1, slice_length = -(-1) = 1, means move [A] to the end
        slice_length = -shift_tensor
    else:
        # For [A,B,C,D], if shift is 1, slice_length = 4 - 1 = 3, means move [A,B,C] to the end
        # The effect equals to move [D] to the beginning
        slice_length = op.Size(self_flatten) - shift_tensor
    # Get second part of the tensor, e.g. [A,B,C]
    suffix = op.Slice(self_flatten, op.Constant(value_ints=[0]), slice_length)
    # Get first part of the tensor, e.g. [D]
    prefix = op.Slice(self_flatten, slice_length, op.Reshape(op.Size(self_flatten), neg_1))
    # Concat first+second together, e.g. [D,A,B,C]
    result = op.Concat(prefix, suffix, axis=0)
    return op.Reshape(result, op.Shape(self))


@torch_op("aten::roll", private=True)
def _aten_roll_shift_and_dim_onnx(self: TTensor, shift: INT64, dim: int) -> TTensor:
    neg_1 = op.Constant(value_ints=[-1])
    dim_tensor = op.Reshape(op.Constant(value_int=dim), neg_1)
    shift_tensor = op.Reshape(shift, neg_1)
    if shift_tensor < 0:
        slice_length = -shift_tensor
    else:
        slice_length = op.Gather(op.Shape(self), dim_tensor, axis=0) - shift_tensor
    # from [A,B,C,D] -> [D,A,B,C], [D] is prefix, [A,B,C] is suffix
    suffix = op.Slice(self, op.Constant(value_ints=[0]), slice_length, axes=dim_tensor)
    prefix = op.Slice(self, slice_length, op.Reshape(op.Size(self), neg_1), axes=dim_tensor)
    result = op.Concat(prefix, suffix, axis=dim)
    return result


def aten_rot90(self: TensorType, k: int = 1, dims: Sequence[int] = (0, 1)) -> TensorType:
    """rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::round")
def aten_round(self: TFloat) -> TFloat:
    """round(Tensor self) -> Tensor"""

    return op.Round(self)


@torch_op("aten::round.decimals")
def aten_round_decimals(self: TFloat, decimals: int = 0) -> TFloat:
    """round.decimals(Tensor self, *, int decimals) -> Tensor"""

    # Scale the input by 10^decimals, round it, and scale it back.
    ten = op.CastLike(10.0, self)
    scale = op.Pow(ten, op.CastLike(decimals, self))
    self_scaled = op.Mul(self, scale)
    rounded = op.Round(self_scaled)
    return op.Div(rounded, scale)


def aten_row_indices(self: TensorType) -> TensorType:
    """row_indices(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


def aten_row_indices_copy(self: TensorType) -> TensorType:
    """row_indices_copy(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_row_stack(tensors: Sequence[TensorType]) -> TensorType:
    """row_stack(Tensor[] tensors) -> Tensor"""

    raise NotImplementedError()


def aten_rrelu(
    self: TensorType,
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
    training: bool = False,
    generator: Optional[str] = None,
) -> TensorType:
    """rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor"""

    raise NotImplementedError()


def aten_rshift(self: TensorType, other: TensorType) -> TensorType:
    """__rshift__.Tensor(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::rsqrt")
def aten_rsqrt(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """rsqrt(Tensor self) -> Tensor"""

    return op.Reciprocal(op.Sqrt(self))


@torch_op(("aten::rsub", "aten::rsub.Scalar"))
def aten_rsub(self: TReal, other: TReal, alpha: float = 1.0) -> TReal:
    """rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    # FIXME(titaiwang): get rid of this when we have type_promotion
    other = op.CastLike(other, self)
    alpha = op.CastLike(alpha, self)
    return op.Sub(other, op.Mul(self, alpha))


@torch_op("aten::scalar_tensor")
def aten_scalar_tensor(s: float, dtype: int = FLOAT.dtype) -> TTensor:  # type: ignore[type-var]
    """scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    return op.Cast(s, to=dtype)


@torch_op("aten::scalar_tensor")
def aten_scalar_tensor_sym_number(
    s: Union[FLOAT, INT32, BOOL], dtype: int = FLOAT.dtype
) -> TTensor:
    """scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    return op.Cast(s, to=dtype)


@torch_op("aten::scatter_add")
def aten_scatter_add(
    self: TReal,
    dim: int,  # we have to use int here because ScatterElements() will use this attribute
    index: TInt,
    src: TReal,
) -> TReal:
    """scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"""

    # if rank(self) == 0 will lead ORT failed, skipped
    return op.ScatterElements(self, index, src, axis=dim, reduction="add")


@torch_op(("aten::scatter_reduce", "aten::scatter_reduce.two"), trace_only=True)
def aten_scatter_reduce(
    self: TReal,
    dim: int,  # we have to use int here because ScatterElements() will use this attribute
    index: TInt,
    src: TReal,
    reduce: str,
    include_self: bool = True,  # pylint: disable=unused-argument
):
    """scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor"""

    reduce_mode = {  # convert torch string name to onnx string name
        "mean": "none",  # 'mean' doesn't support in ONNX 1.14 definition
        "sum": "add",
        "prod": "mul",
        "amin": "min",
        "amax": "max",
    }
    onnx_reduce = reduce_mode[reduce]
    return _aten_scatter_reduce_onnx(self, index, src, dim, onnx_reduce)


@torch_op("aten::scatter_reduce", private=True)
def _aten_scatter_reduce_onnx(
    self: TReal,
    index: TInt,
    src: TReal,
    dim: int,
    onnx_reduce: str,
):
    self_rank = op.Size(op.Shape(self))
    if self_rank == 0:  # assert (index_rank == 0 and rank_src == 0)
        neg_1 = op.Constant(value_ints=[-1])
        self = op.Reshape(self, neg_1)
        index = op.Reshape(index, neg_1)
        src = op.Reshape(src, neg_1)
    result = op.ScatterElements(self, index, src, axis=dim, reduction=onnx_reduce)
    if self_rank == 0:
        result = op.Squeeze(result)
    return result


def aten_searchsorted(
    sorted_sequence: TensorType,
    self: TensorType,
    out_int32: bool = False,
    right: bool = False,
    side: Optional[str] = None,
    sorter: Optional[TensorType] = None,
) -> TensorType:
    """searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor"""

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
    """segment_reduce(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, Tensor? offsets=None, int axis=0, bool unsafe=False, Scalar? initial=None) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::select", "aten::select.int"))
def aten_select(self: TTensor, dim: int, index: int) -> TTensor:
    """select(Tensor self, int dim, int index) -> Tensor"""

    return op.Gather(self, index, axis=dim)


def aten_select_backward(
    grad_output: TensorType, input_sizes: INT64, dim: int, index: int
) -> TensorType:
    """select_backward(Tensor grad_output, SymInt[] input_sizes, int dim, int index) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::select_scatter")
def aten_select_scatter(self: TensorType, src: TensorType, dim: int, index: int) -> TensorType:
    """select_scatter(Tensor self, Tensor src, int dim, int index) -> Tensor"""

    # Change src rank to self rank according to dim
    # e.g. if self is [2,3,4], src is [2,4], dim=1, then update is [2,1,4]
    update = op.Unsqueeze(src, axes=dim)
    # Change index rank to the same as 'update' [2,1,4]
    indices = op.Expand(index, op.Shape(update))
    return op.ScatterElements(self, indices, update, axis=dim, reduction="none")


@torch_op("aten::selu")
def aten_selu(self: TFloat) -> TFloat:
    """selu(Tensor self) -> Tensor"""

    return op.Selu(self)


def aten_set_data(self: TensorType, new_data: TensorType) -> Any:
    """set_data(Tensor(a!) self, Tensor new_data) -> ()"""

    raise NotImplementedError()


def aten_sgn(self: TensorType) -> TensorType:
    """sgn(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::sigmoid")
def aten_sigmoid(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """sigmoid(Tensor self) -> Tensor"""

    return op.Sigmoid(self)


@torch_op("aten::sign")
def aten_sign(self: TReal) -> TReal:
    """sign(Tensor self) -> Tensor"""

    return op.Sign(self)


def aten_signbit(self: TensorType) -> TensorType:
    """signbit(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::sin")
def aten_sin(self: TFloat) -> TFloat:
    """sin(Tensor self) -> Tensor"""

    return op.Sin(self)


@torch_op("aten::sinh")
def aten_sinh(self: TFloat) -> TFloat:
    """sinh(Tensor self) -> Tensor"""

    return op.Sinh(self)


@torch_op(("aten::slice", "aten::slice.Tensor"), trace_only=True)
def aten_slice(
    self: TTensor,
    dim: int = 0,
    start: Optional[INT64] = None,
    end: Optional[INT64] = None,
    step: Optional[INT64] = None,
) -> TTensor:
    """slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)"""

    # TODO: using OptionalHasElement() to check start/end value
    if start is not None:
        start = op.Cast(start, to=INT64.dtype)
        start = op.Reshape(start, op.Constant(value_ints=[-1]))
    else:
        start = op.Constant(value_ints=[0])

    if end is not None:
        end = op.Cast(end, to=INT64.dtype)
        end = op.Reshape(end, op.Constant(value_ints=[-1]))
    else:
        end = op.Constant(value_ints=[_INT64_MAX])

    dim = op.Cast(dim, to=INT64.dtype)
    dim = op.Reshape(dim, op.Constant(value_ints=[-1]))

    if step is not None:
        step = op.Cast(step, to=INT64.dtype)
        step = op.Reshape(step, op.Constant(value_ints=[-1]))
    else:
        step = op.Constant(value_ints=[1])

    return op.Slice(self, start, end, dim, step)


def aten_slice_backward(
    grad_output: TensorType,
    input_sizes: INT64,
    dim: int,
    start: INT64,
    end: INT64,
    step: INT64,
) -> TensorType:
    """slice_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step) -> Tensor"""

    raise NotImplementedError()


def aten_slice_copy(
    self: TensorType,
    dim: int = 0,
    start: Optional[INT64] = None,
    end: Optional[INT64] = None,
    step: INT64 = 1,
) -> TensorType:
    """slice_copy.Tensor(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::slice_scatter")
def aten_slice_scatter(
    self: TTensor,
    src: TTensor,
    dim: int = 0,
    start: Optional[INT64] = None,
    end: Optional[INT64] = None,
    step: INT64 = 1,
) -> TTensor:
    """slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor"""

    # Although 'start' and 'end' can be None in signature, but actually 'start' must be specified
    # Assert(start is not None)
    # And, 'end' also must be specified, and end-start must be equal to the size of 'src'
    # Assert(end-start == shape(src) > 0)
    # Try torch sample to get more information:
    # https://pytorch.org/docs/master/generated/torch.slice_scatter.html?highlight=slice_scatter#torch.slice_scatter
    # Take (torch.zeros(8, 8), torch.ones(2, 8), 0, 6, 64, 1) as example:
    # Step 1: get 1D tensor from 0 to dim_size-1, then Slice it using start, end and step.
    # We cannot use Range(start, end, step) directly as start or end may out of range.
    # For the example, the output of this step is Slice([0, ..., 7], 6, 64, 1) = [6, 7]
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    self_shape = op.Shape(self)
    dim_shape = op.Gather(self_shape, dim, axis=0)
    index_base = op.Range(0, dim_shape, 1)
    index_base = op.Slice(
        index_base,
        op.Unsqueeze(start, zero),
        op.Unsqueeze(end, zero),
        zero,
        op.Unsqueeze(step, zero),
    )
    # Step 2: Unsqueeze to add 1s preparing for Expand.
    # Need to handle negative dim here.
    # For the example above, the result of this step is [[6],[7]].
    index_base = op.Unsqueeze(
        index_base, op.Range(1, op.Where(dim < 0, 0, op.Size(self_shape)) - dim, 1)
    )
    # Step 3: Expand the indices.
    # For the example above, it's Expand([[6],[7]], (1, 8)) = [[6,...,6],[7,...,7]].
    shape_expand = op.ScatterElements(
        self_shape, op.Unsqueeze(op.Constant(value_int=dim), zero), one, axis=0
    )
    indices = op.Expand(index_base, shape_expand)
    # Step 4: final ScatterElements.
    return op.ScatterElements(self, indices, src, axis=dim)


def aten_slogdet(self: TensorType) -> tuple[TensorType, TensorType]:
    """slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)"""

    raise NotImplementedError()


def aten_smm(self: TensorType, mat2: TensorType) -> TensorType:
    """smm(Tensor self, Tensor mat2) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::softmax", "aten::softmax.int", "aten::special_softmax"))
def aten_softmax(
    self: TFloatOrBFloat16, dim: int, dtype: int = FLOAT.dtype
) -> TFloatOrBFloat16:
    """softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"""

    self_is_scalar = op.Size(op.Shape(self)) == 0
    if self_is_scalar:
        self = op.Unsqueeze(self, op.Constant(value_ints=[0]))
    result = op.Softmax(self, axis=dim)
    result = op.Cast(result, to=dtype)
    if self_is_scalar:
        # Convert to scalar when input is scalar
        result = op.Squeeze(result)

    return result


@torch_op(("aten::softmax", "aten::softmax.int", "aten::special_softmax"))
def aten_softmax_no_dtype(self: TFloatOrBFloat16, dim: int) -> TFloatOrBFloat16:
    """softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"""

    self_is_scalar = op.Size(op.Shape(self)) == 0
    if self_is_scalar:
        self = op.Unsqueeze(self, op.Constant(value_ints=[0]))
    result = op.Softmax(self, axis=dim)
    if self_is_scalar:
        # Convert to scalar when input is scalar
        result = op.Squeeze(result)

    return result


def aten_sort(
    self: TensorType, dim: int = -1, descending: bool = False
) -> tuple[TensorType, TensorType]:
    """sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)"""

    raise NotImplementedError()


def aten_sparse_dim(self: TensorType) -> int:
    """sparse_dim(Tensor self) -> int"""

    raise NotImplementedError()


def aten_sparse_mask(self: TensorType, mask: TensorType) -> TensorType:
    """sparse_mask(Tensor self, Tensor mask) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::split", "aten::split.Tensor"))
def aten_split(self: TTensor, split_size: INT64, dim: int = 0) -> TTensor:
    """split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]"""

    return op.SplitToSequence(self, split_size, axis=dim)


def aten_split_copy(self: TensorType, split_size: INT64, dim: int = 0) -> TensorType:
    """split_copy.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]"""

    raise NotImplementedError()


@torch_op("aten::split_with_sizes")
def aten_split_with_sizes(self: TTensor, split_sizes: INT64, dim: int = 0) -> TTensor:
    """split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]"""

    return op.SplitToSequence(self, split_sizes, axis=dim)


def aten_split_with_sizes_copy(
    self: TensorType, split_sizes: INT64, dim: int = 0
) -> TensorType:
    """split_with_sizes_copy(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]"""

    raise NotImplementedError()


@torch_op("aten::sqrt")
def aten_sqrt(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """sqrt(Tensor self) -> Tensor"""

    return op.Sqrt(self)


def aten_square(self: TensorType) -> TensorType:
    """square(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::squeeze")
def aten_squeeze(self: TTensor) -> TTensor:
    """squeeze(Tensor(a) self) -> Tensor(a)"""

    return op.Squeeze(self)


@torch_op("aten::squeeze.dim")
def aten_squeeze_dim(self: TTensor, dim: int) -> TTensor:
    result = self
    if op.Size(op.Shape(self)) > 0:  # type: ignore[operator]
        # check if specified dimension is 1, do squeeze
        shape = op.Shape(self)
        dim_size = op.Gather(shape, dim, axis=0)
        if dim_size == 1:
            dims = op.Reshape(dim, op.Constant(value_ints=[-1]))
            result = op.Squeeze(self, dims)

    return result


def aten_squeeze_copy(self: TensorType) -> TensorType:
    """squeeze_copy(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_sspaddmm(
    self: TensorType, mat1: TensorType, mat2: TensorType, beta: float = 1.0, alpha: float = 1.0
) -> TensorType:
    """sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::stack")
def aten_stack(tensors: Sequence[TTensorOrString], dim: int = 0) -> TTensorOrString:
    """stack(Tensor[] tensors, int dim=0) -> Tensor"""
    return op.ConcatFromSequence(tensors, axis=dim, new_axis=1)


def aten_std(self: TensorType, unbiased: bool = True) -> TensorType:
    """std(Tensor self, bool unbiased=True) -> Tensor"""

    raise NotImplementedError()


def aten_std_mean(self: TensorType, unbiased: bool = True) -> tuple[TensorType, TensorType]:
    """std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)"""

    raise NotImplementedError()


@torch_op("aten::stft", private=True)
def _add_batch_dimension(self: TFloatOrBFloat16) -> Tuple[TFloatOrBFloat16, INT64]:
    signal_shape = op.Shape(self)
    signal_rank = op.Size(signal_shape)
    if signal_rank == 1:
        # Add a batch dimension
        self = op.Unsqueeze(self, op.Constant(value_ints=[0]))
    return self, signal_rank


@torch_op("aten::stft", private=True)
def _center_window_around_zeros_if_needed(
    window: TFloatOrBFloat16, n_fft: int
) -> TFloatOrBFloat16:
    # first dimension
    n_win = op.Gather(op.Shape(window), 0)
    # Center window around zeros if needed (required by ONNX's STFT)
    if n_win < n_fft:
        left = (n_fft - n_win) / 2

        right = n_fft - left - n_win
        left = op.Reshape(left, op.Constant(value_ints=[1]))
        right = op.Reshape(right, op.Constant(value_ints=[1]))

        left_win = op.Expand(op.Constant(value_ints=[0]), left)
        right_win = op.Expand(op.Constant(value_ints=[0]), right)
        right_win = op.CastLike(right_win, window)
        left_win = op.CastLike(left_win, window)
        window = op.Concat(left_win, window, right_win, axis=0)
    return window


@torch_op("aten::stft", private=True)
def _create_window_from_win_length(win_length: int, n_fft: int) -> TFloatOrBFloat16:
    left = (n_fft - win_length) / 2

    right = n_fft - left - win_length
    left = op.Reshape(left, op.Constant(value_ints=[1]))
    right = op.Reshape(right, op.Constant(value_ints=[1]))
    win_length = op.Reshape(win_length, op.Constant(value_ints=[1]))

    left_win = op.Expand(op.Constant(value_ints=[0]), left)
    right_win = op.Expand(op.Constant(value_ints=[0]), right)
    window_list = op.Expand(op.Constant(value_ints=[1]), win_length)
    return op.Concat(left_win, window_list, right_win, axis=0)


@torch_op("aten::stft", private=True)
def _create_window_from_n_fft(n_fft: int) -> TFloatOrBFloat16:
    n_fft_tensor = op.Reshape(n_fft, op.Constant(value_ints=[1]))
    window = op.Expand(op.Constant(value_ints=[1]), n_fft_tensor)
    return window


@torch_op("aten::stft", private=True)
def _normalize_fft_result(
    signal: TFloatOrBFloat16, result: TFloatOrBFloat16, n_fft: int
) -> TFloatOrBFloat16:
    n_fft_tensor = op.Reshape(n_fft, op.Constant(value_ints=[1]))
    sqrt_nfft = op.Sqrt(op.CastLike(n_fft_tensor, signal))
    result = result / sqrt_nfft
    return result


@torch_op("aten::stft", private=True)
def _aten_stft_onnx(
    signal: TFloatOrBFloat16,
    frame_step_const: INT64,
    window: Union[TFloatOrBFloat16, INT64],
    frame_length_const: INT64,
    signal_rank: INT64,
    onesided: int,
) -> TFloatOrBFloat16:
    window = op.CastLike(window, signal)
    result = op.STFT(signal, frame_step_const, window, frame_length_const, onesided=onesided)
    result = op.Transpose(result, perm=[0, 2, 1, 3])
    # Remove batch dimension, if needed
    if signal_rank == 1:
        result = op.Squeeze(result, op.Constant(value_ints=[0]))
    return result


@torch_op("aten::stft", trace_only=True)
def aten_stft(
    self: TFloatOrBFloat16,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[TFloatOrBFloat16] = None,
    normalized: bool = False,
    onesided: Optional[bool] = None,
    return_complex: Optional[bool] = None,
) -> TFloatOrBFloat16:
    """stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor"""

    # NOTE: regarless of the value of return_complex, we always return a real representation.
    del return_complex

    # Get STFT sizes
    if hop_length is None:
        # core dump
        # hop_leagth = op.Div(op.Constant(value_ints=n_fft), op.Constant(value_ints=[4]))
        hop_length = n_fft // 4
    frame_step_const = op.Reshape(hop_length, op.Constant(value_ints=[1]))
    frame_length_const = op.Reshape(n_fft, op.Constant(value_ints=[1]))

    # Pre-process input if needed
    self, signal_rank = _add_batch_dimension(self)

    # Get window and make sure it's the same size as `win_length` or `n_fft`
    if window is not None and window.shape[0] is not None:
        window = _center_window_around_zeros_if_needed(window, n_fft)
    elif window is None:
        if win_length is not None:
            window = _create_window_from_win_length(win_length, n_fft)
        else:
            window = _create_window_from_n_fft(n_fft)

    if onesided is None or onesided:
        onesided = 1
    else:
        onesided = 0
    # remove batch dimension included
    result = _aten_stft_onnx(
        self, frame_step_const, window, frame_length_const, signal_rank, onesided
    )

    # Normalize, if needed
    if normalized:
        result = _normalize_fft_result(self, result, n_fft)

    return result


@torch_op(("aten::sub", "aten::sub.Tensor"))
def aten_sub(self: TReal, other: TReal, alpha: float = 1.0) -> TReal:
    """sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    alpha = op.CastLike(alpha, other)
    other = op.Mul(other, alpha)

    return op.Sub(self, other)


def aten_subtract(self: TensorType, other: TensorType, alpha: float = 1.0) -> TensorType:
    """subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::sum", "aten::sum.dim_IntList"), trace_only=True)
def aten_sum_dim_IntList(
    self: TReal, dim: Optional[INT64] = None, keepdim: bool = False, dtype: int = -1
) -> TReal:
    """sum(Tensor self, SymInt dim, bool keepdim, *, ScalarType? dtype=None) -> Tensor"""

    # NOTE: trace_only because both if branches need to be the same type, but we have
    # a cast in the if branch.

    # TODO: Combine the overloads when OptionalHasElement() works
    if dim is None:
        result = _aten_sum_dim_none(self, keepdim=keepdim)
    else:
        result = _aten_sum_dim_onnx(self, dim, keepdim=keepdim)

    if dtype != -1:
        result = op.Cast(result, to=dtype)

    return result


@torch_op("aten::sum", private=True)
def _aten_sum_dim_onnx(self: TReal, dim: INT64, keepdim: bool = False) -> TReal:
    self_is_scalar = op.Size(op.Shape(self)) == 0
    if self_is_scalar:
        self = op.Reshape(self, op.Constant(value_ints=[-1]))

    if op.Size(op.Shape(dim)) == 0:
        dim = op.Reshape(dim, op.Constant(value_ints=[-1]))
        dim = op.Cast(dim, to=INT64.dtype)
    result = op.ReduceSum(self, dim, keepdims=keepdim)

    if self_is_scalar:
        result = op.Squeeze(result)
    return result


@torch_op("aten::sum", private=True)
def _aten_sum_dim_none(self: TReal, keepdim: bool = False) -> TReal:
    self_is_scalar = op.Size(op.Shape(self)) == 0
    if self_is_scalar:
        self = op.Reshape(self, op.Constant(value_ints=[-1]))

    result = op.ReduceSum(self, keepdims=keepdim)

    if self_is_scalar:
        result = op.Squeeze(result)
    return result


def aten_sum_to_size(self: TensorType, size: Sequence[int]) -> TensorType:
    """sum_to_size(Tensor self, int[] size) -> Tensor"""

    raise NotImplementedError()


def aten_svd(
    self: TensorType, some: bool = True, compute_uv: bool = True
) -> tuple[TensorType, TensorType, TensorType]:
    """svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)"""

    raise NotImplementedError()


def aten_swapaxes(self: TensorType, axis0: int, axis1: int) -> TensorType:
    """swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)"""

    raise NotImplementedError()


def aten_swapdims(self: TensorType, dim0: int, dim1: int) -> TensorType:
    """swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)"""

    raise NotImplementedError()


@torch_op("aten::sym_size")
def aten_sym_size(self: TReal, dim: int = 0) -> TReal:
    """sym_size(Tensor self, int dim) -> Tensor"""
    # NOTE: onnxscript doesn't support attribute process,
    # so op.Shape(self, start=dim, end=dim + 1) is not supported.

    # TODO(titaiwang): ORT==1.15 fixes SegFault
    # https://github.com/microsoft/onnxscript/pull/484#discussion_r1136105039
    # Change the op to:
    # shape = op.Shape(self)
    # idx= op.Reshape(dim, [1])
    # return op.Gather(shape, idx)

    shape = op.Shape(self)
    # Reshape helps dim from int to tensor, and
    # input arguments support attribute processing.
    start = op.Reshape(dim, op.Constant(value_ints=[1]))
    end = op.Reshape(dim + 1, op.Constant(value_ints=[1]))
    return op.Slice(shape, start, end)


def aten_symeig(
    self: TensorType, eigenvectors: bool = False, upper: bool = True
) -> tuple[TensorType, TensorType]:
    """symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)"""

    raise NotImplementedError()


@torch_op("aten::t")
def aten_t(self: TTensor) -> TTensor:
    """t(Tensor(a) self) -> Tensor(a)"""

    rank = op.Size(op.Shape(self))
    if rank == 2:
        result = op.Transpose(self, perm=[1, 0])
    else:
        # rank < 2
        result = self
    return result


def aten_t_copy(self: TensorType) -> TensorType:
    """t_copy(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_take(self: TensorType, index: TensorType) -> TensorType:
    """take(Tensor self, Tensor index) -> Tensor"""

    raise NotImplementedError()


def aten_take_along_dim(
    self: TensorType, indices: TensorType, dim: Optional[int] = None
) -> TensorType:
    """take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::tan")
def aten_tan(self: TFloat) -> TFloat:
    """tan(Tensor self) -> Tensor"""

    return op.Tan(self)


@torch_op("aten::tanh")
def aten_tanh(self: TFloat) -> TFloat:
    """tanh(Tensor self) -> Tensor"""

    return op.Tanh(self)


@torch_op("aten::tensor.bool")
def aten_tensor_bool(self: bool, dtype: int) -> TensorType:
    tensor = op.Constant(value_int=self)
    return op.Cast(tensor, to=dtype)


@torch_op("aten::tensor.float")
def aten_tensor_float(self: float, dtype: int) -> TensorType:
    tensor = op.Constant(value_float=self)
    return op.Cast(tensor, to=dtype)


@torch_op("aten::tensor.int")
def aten_tensor_int(self: int, dtype: int) -> TensorType:
    tensor = op.Constant(value_int=self)
    return op.Cast(tensor, to=dtype)


def aten_tensordot(
    self: TensorType, other: TensorType, dims_self: Sequence[int], dims_other: Sequence[int]
) -> TensorType:
    """tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor"""

    raise NotImplementedError()


def aten_threshold(self: TensorType, threshold: float, value: float) -> TensorType:
    """threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor"""

    raise NotImplementedError()


def aten_threshold_backward(
    grad_output: TensorType, self: TensorType, threshold: float
) -> TensorType:
    """threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::tile")
def aten_tile(self: TTensor, dims: INT64) -> TTensor:
    """tile(Tensor self, int[] dims) -> Tensor"""

    self_shape = op.Shape(self)
    self_rank = op.Size(self_shape)
    dims_rank = op.Size(dims)
    diff = op.Sub(self_rank, dims_rank)

    if diff > 0:
        # dims is shorter than self.shape
        # pad dims with 1
        diff_1d = op.Reshape(diff, op.Constant(value_ints=[1]))
        exapnd_ones = op.Expand(op.Constant(value_ints=[1]), diff_1d)
        dims = op.Concat(exapnd_ones, dims, axis=0)

    if diff < 0:
        # dims is longer than self.shape
        # pad self.shape with 1
        diff_1d = op.Reshape(op.Abs(diff), op.Constant(value_ints=[1]))
        exapnd_ones = op.Expand(op.Constant(value_ints=[1]), diff_1d)
        self_final_shape = op.Concat(exapnd_ones, self_shape, axis=0)
        self = op.Reshape(self, self_final_shape)

    return op.Tile(self, dims)


def aten_to_dense(self: TensorType, dtype: Optional[int] = None) -> TensorType:
    """to_dense(Tensor self, ScalarType? dtype=None) -> Tensor"""

    raise NotImplementedError()


def aten_to_dense_backward(grad: TensorType, input: TensorType) -> TensorType:
    """to_dense_backward(Tensor grad, Tensor input) -> Tensor"""

    raise NotImplementedError()


def aten_to_mkldnn(self: TensorType, dtype: Optional[int] = None) -> TensorType:
    """to_mkldnn(Tensor self, ScalarType? dtype=None) -> Tensor"""

    raise NotImplementedError()


def aten_to_mkldnn_backward(grad: TensorType, input: TensorType) -> TensorType:
    """to_mkldnn_backward(Tensor grad, Tensor input) -> Tensor"""

    raise NotImplementedError()


def aten_to_padded_tensor(
    self: TensorType, padding: float, output_size: Optional[INT64] = None
) -> TensorType:
    """to_padded_tensor(Tensor self, float padding, SymInt[]? output_size=None) -> Tensor"""

    raise NotImplementedError()


def aten_to_sparse(self: TensorType) -> TensorType:
    """to_sparse(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_to_sparse_bsc(self: TensorType, blocksize: Sequence[int]) -> TensorType:
    """to_sparse_bsc(Tensor self, int[2] blocksize) -> Tensor"""

    raise NotImplementedError()


def aten_to_sparse_bsr(self: TensorType, blocksize: Sequence[int]) -> TensorType:
    """to_sparse_bsr(Tensor self, int[2] blocksize) -> Tensor"""

    raise NotImplementedError()


def aten_to_sparse_csc(self: TensorType) -> TensorType:
    """to_sparse_csc(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_to_sparse_csr(self: TensorType) -> TensorType:
    """to_sparse_csr(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::topk")
def aten_topk(
    self: TReal, k: INT64, dim: int = -1, largest: bool = True, sorted: bool = True
) -> Tuple[TReal, INT64]:
    """topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)"""

    self_is_scalar = op.Size(op.Shape(self)) == 0
    if self_is_scalar:
        self = op.Unsqueeze(self, op.Constant(value_ints=[0]))
    k = op.Reshape(op.Cast(k, to=INT64.dtype), op.Constant(value_ints=[1]))
    values, indices = op.TopK(self, k, axis=dim, largest=largest, sorted=sorted)
    if self_is_scalar:
        values = op.Squeeze(values, op.Constant(value_ints=[0]))
        indices = op.Squeeze(indices, op.Constant(value_ints=[0]))
    return values, indices


def aten_trace(self: TensorType) -> TensorType:
    """trace(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_trace_backward(grad: TensorType, sizes: INT64) -> TensorType:
    """trace_backward(Tensor grad, SymInt[] sizes) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::transpose", "aten::transpose.int"), trace_only=True)
def aten_transpose(self, dim0: int, dim1: int):
    """transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)"""

    # Use trace only to construct the prem attribute in Transpose
    self_rank = len(self.shape)  # type: ignore[attr-defined]

    if self_rank == 0:
        result = self
    else:
        # Python code, change when onnxscript supports this
        dims = list(range(self_rank))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        # Python code ends

        result = op.Transpose(self, perm=dims)

    return result


def aten_triangular_solve(
    self: TensorType,
    A: TensorType,
    upper: bool = True,
    transpose: bool = False,
    unitriangular: bool = False,
) -> tuple[TensorType, TensorType]:
    """triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)"""

    raise NotImplementedError()


@torch_op("aten::tril")
def aten_tril(self: TTensor, diagonal: int = 0) -> TTensor:
    """tril(Tensor self, int diagonal=0) -> Tensor"""

    return op.Trilu(self, diagonal, upper=0)


def aten_tril_indices(row: int, col: int, offset: int = 0) -> TensorType:
    """tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


def aten_triplet_margin_loss(
    anchor: TensorType,
    positive: TensorType,
    negative: TensorType,
    margin: float = 1.0,
    p: float = 2.0,
    eps: float = 1e-06,
    swap: bool = False,
    reduction: int = 1,
) -> TensorType:
    """triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::triu")
def aten_triu(self: TensorType, diagonal: int = 0) -> TensorType:
    """triu(Tensor self, int diagonal=0) -> Tensor"""

    return op.Trilu(self, diagonal, upper=1)


def aten_triu_indices(row: int, col: int, offset: int = 0) -> TensorType:
    """triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::trunc")
def aten_trunc(self: TFloatOrBFloat16) -> TFloatOrBFloat16:
    """trunc(Tensor self) -> Tensor"""

    # Reference https://github.com/onnx/onnx/issues/4588#issuecomment-1463970126
    integer_parts = op.Floor(op.Abs(self))
    is_negative = op.Less(self, 0.0)
    return op.Where(is_negative, op.Neg(integer_parts), integer_parts)


def aten_type_as(self: TensorType, other: TensorType) -> TensorType:
    """type_as(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::unbind", "aten::unbind.int"))
def aten_unbind(self: TTensor, dim: int = 0) -> Sequence[TTensor]:
    """unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]"""

    split_sizes = op.Constant(value_int=1)
    return op.SplitToSequence(self, split_sizes, axis=dim, keepdims=False)


@torch_op("aten::unflatten")
def aten_unflatten(self: TReal, dim: INT64, sizes: INT64):
    """unflatten(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)"""

    self_size = op.Shape(self)

    # PyTorch accepts negative dim as reversed counting
    self_rank = op.Size(self_size)
    dim = self_rank + dim
    dim = dim % self_rank

    head_start_idx = op.Constant(value_ints=[0])
    head_end_idx = op.Reshape(dim, op.Constant(value_ints=[1]))
    head_part_rank = op.Slice(self_size, head_start_idx, head_end_idx)

    tail_start_idx = op.Reshape(dim + 1, op.Constant(value_ints=[1]))
    tail_end_idx = op.Constant(value_ints=[_INT64_MAX])
    tail_part_rank = op.Slice(self_size, tail_start_idx, tail_end_idx)

    final_shape = op.Concat(head_part_rank, sizes, tail_part_rank, axis=0)

    return op.Reshape(self, final_shape)


@torch_op("aten::unfold", trace_only=True)
def aten_unfold(self: TTensor, dimension: int, size: int, step: int) -> TTensor:
    """unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)"""

    self_rank = len(self.shape)
    if self_rank == 0:
        result = op.Unsqueeze(self, 0)
    else:
        dim_size = self.shape[dimension]
        target_end = (dim_size - size) // step + 1
        if target_end >= 1:  # the rank of final reuslt will be self_rank + 1
            self_rank = self_rank + 1
        # perm need to be list[int], so have to be generated in trace_only mode
        perm = list(range(self_rank))
        # from [0,1,2,3,4] -> [0,1,3,4,2] when dimension=1
        perm.append(perm.pop(dimension + 1))
        result = _aten_unfold_onnx(self, dimension, size, step, target_end, perm)
    return result


@torch_op("aten::unfold", private=True)
def _aten_unfold_onnx(
    self: TTensor, dim: int, size: int, step: int, target_end: int, perm: Sequence[int]
) -> TTensor:
    dims = op.Reshape(op.Constant(value_int=dim), op.Constant(value_ints=[-1]))
    # FIXME: the dtype for this function cannot work, default to float
    seq_result = op.SequenceEmpty()
    i = op.Constant(value_ints=[0])
    cond = i < target_end
    while cond:  # because for loop cannot work here, so use while loop
        starts = i * step  # starts is [0, step, step*2, step*3, ...]
        ends = starts + size  # ends is [0+size, step+size, step*2+size, step*3+size, ...]
        slice_result = op.Slice(self, starts, ends, dims)
        # sequence only support float32
        slice_result_float32 = op.Cast(slice_result, to=FLOAT.dtype)
        seq_result = op.SequenceInsert(seq_result, slice_result_float32)
        i = i + 1
        cond = i < target_end
    concat_result = op.ConcatFromSequence(seq_result, axis=dim, new_axis=1)
    result = op.Transpose(concat_result, perm=perm)
    return op.CastLike(result, self)


def aten_unfold_backward(
    grad_in: TensorType, input_sizes: INT64, dim: int, size: int, step: int
) -> TensorType:
    """unfold_backward(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step) -> Tensor"""

    raise NotImplementedError()


def aten_unfold_copy(self: TensorType, dimension: int, size: int, step: int) -> TensorType:
    """unfold_copy(Tensor self, int dimension, int size, int step) -> Tensor"""

    raise NotImplementedError()


def aten_unique_consecutive(
    self: TensorType,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: Optional[int] = None,
) -> tuple[TensorType, TensorType, TensorType]:
    """unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)"""

    raise NotImplementedError()


def aten_unique_dim(
    self: TensorType,
    dim: int,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple[TensorType, TensorType, TensorType]:
    """unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)"""

    raise NotImplementedError()


def aten_unique_dim_consecutive(
    self: TensorType, dim: int, return_inverse: bool = False, return_counts: bool = False
) -> tuple[TensorType, TensorType, TensorType]:
    """unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)"""

    raise NotImplementedError()


def aten_unsafe_chunk(self: TensorType, chunks: int, dim: int = 0) -> TensorType:
    """unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]"""

    raise NotImplementedError()


def aten_unsafe_split(self: TensorType, split_size: INT64, dim: int = 0) -> TensorType:
    """unsafe_split.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]"""

    raise NotImplementedError()


def aten_unsafe_split_with_sizes(
    self: TensorType, split_sizes: INT64, dim: int = 0
) -> TensorType:
    """unsafe_split_with_sizes(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]"""

    raise NotImplementedError()


@torch_op("aten::unsqueeze")
def aten_unsqueeze(self: TTensor, dim: int) -> TTensor:
    """unsqueeze(Tensor(a) self, int dim) -> Tensor(a)"""

    dim = op.Cast(dim, to=INT64.dtype)
    return op.Unsqueeze(self, dim)


def aten_unsqueeze_copy(self: TensorType, dim: int) -> TensorType:
    """unsqueeze_copy(Tensor self, int dim) -> Tensor"""

    raise NotImplementedError()


def aten_value_selecting_reduction_backward(
    grad: TensorType, dim: int, indices: TensorType, sizes: INT64, keepdim: bool
) -> TensorType:
    """value_selecting_reduction_backward(Tensor grad, int dim, Tensor indices, SymInt[] sizes, bool keepdim) -> Tensor"""

    raise NotImplementedError()


def aten_values(self: TensorType) -> TensorType:
    """values(Tensor(a) self) -> Tensor(a)"""

    raise NotImplementedError()


def aten_values_copy(self: TensorType) -> TensorType:
    """values_copy(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_vander(
    x: TensorType, N: Optional[int] = None, increasing: bool = False
) -> TensorType:
    """vander(Tensor x, int? N=None, bool increasing=False) -> Tensor"""

    raise NotImplementedError()


def aten_var(self: TensorType, unbiased: bool = True) -> TensorType:
    """var(Tensor self, bool unbiased=True) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::var_mean", trace_only=True)
def aten_var_mean(self: TReal, unbiased: bool = True) -> Tuple[TReal, TReal]:
    """var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)"""

    # Assume bool(True) and int(1) are same in ONNX, so pass "unbiased" directly as "correction"
    # If not this case, should be explicitly set correction value according to unbiased value
    return _aten_var_mean_onnx(self, correction=float(unbiased), keepdim=False)


@torch_op("aten::var_mean.dim", trace_only=True)
def aten_var_mean_dim(
    self: TReal, dim: Optional[int], unbiased: bool = True, keepdim: bool = False
) -> Tuple[TReal, TReal]:
    """var_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)"""

    # Although dim is Optional in signature, but we assume it must has value for this overload
    # Assert(dim is not None)
    if isinstance(dim, Tuple):
        dim_tensor = op.Constant(value_ints=dim)
    else:
        dim_tensor = op.Constant(value_int=dim)
    return _aten_var_mean_dim_onnx(
        self, dim_tensor, correction=float(unbiased), keepdim=keepdim
    )


@torch_op("aten::var_mean.correction", trace_only=True)
def aten_var_mean_correction(
    self: TReal,
    dim: Optional[int] = None,
    correction: Optional[float] = None,
    keepdim: bool = False,
) -> Tuple[TReal, TReal]:
    """var_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)"""

    if correction is None:
        correction = 1.0

    if dim is None:
        var, mean = _aten_var_mean_onnx(self, correction, keepdim)
    else:
        if isinstance(dim, Tuple):
            dim_tensor = op.Constant(value_ints=dim)
        else:
            dim_tensor = op.Constant(value_int=dim)
        var, mean = _aten_var_mean_dim_onnx(self, dim_tensor, correction, keepdim)
    return var, mean


@torch_op("aten::var_mean", private=True)
def _aten_var_mean_onnx(
    self: TReal, correction: float = 1.0, keepdim: bool = False
) -> Tuple[TReal, TReal]:
    # Compute mean and var
    mean = op.ReduceMean(self, keepdims=keepdim)
    sub_mean = op.Sub(self, mean)
    sqr_mean = op.Mul(sub_mean, sub_mean)
    var = op.ReduceMean(sqr_mean, keepdims=keepdim)
    # Adjust var according to correction value
    if correction > 0.0:
        self_shape = op.Shape(self)
        numel_float = op.CastLike(op.ReduceProd(self_shape, keepdims=False), self)
        mul = op.Mul(var, numel_float)
        sub = op.Sub(numel_float, op.CastLike(correction, self))
        var = op.Div(mul, sub)

    return var, mean


@torch_op("aten::var_mean.dim", private=True)
def _aten_var_mean_dim_onnx(
    self: TReal, dim: INT64, correction: float, keepdim: bool = False
) -> Tuple[TReal, TReal]:
    dim = op.Reshape(dim, op.Constant(value_ints=[-1]))
    # Computer mean and var
    mean = op.ReduceMean(self, dim, keepdims=keepdim)
    sub_mean = op.Sub(self, op.ReduceMean(self, dim, keepdims=True))
    sqr_mean = op.Mul(sub_mean, sub_mean)
    var = op.ReduceMean(sqr_mean, dim, keepdims=keepdim)
    # Adjust var according to correction value
    if correction > 0.0:
        self_shape = op.Shape(self)
        dim_size = op.Gather(self_shape, dim, axis=0)
        numel_float = op.CastLike(op.ReduceProd(dim_size, keepdims=False), self)
        mul = op.Mul(var, numel_float)
        sub = op.Sub(numel_float, correction)
        var = op.Div(mul, sub)

    return var, mean


def aten_vdot(self: TensorType, other: TensorType) -> TensorType:
    """vdot(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::view")
def aten_view(self: TTensor, size: IntType) -> TTensor:
    """view(Tensor(a) self, SymInt[] size) -> Tensor(a)"""

    size = op.Cast(size, to=INT64.dtype)  # Reshape only support INT64 as second input
    return op.Reshape(self, size)


@torch_op("aten::view_as")
def aten_view_as(self: TTensor, other: TTensor2) -> TTensor:
    """view_as(Tensor(a) self, Tensor other) -> Tensor(a)"""

    size = op.Shape(other)
    return op.Reshape(self, size)


@torch_op("aten::view_as_complex")
def aten_view_as_complex(self: TTensor) -> TTensor:
    """view_as_complex(Tensor(a) self) -> Tensor(a)"""

    # We always operate on the real representation of a complex number in torchlib
    # So this is a no-op
    return op.Identity(self)


@torch_op("aten::view_as_complex_copy")
def aten_view_as_complex_copy(self: TTensor) -> TTensor:
    """view_as_complex_copy(Tensor self) -> Tensor"""

    # We always operate on the real representation of a complex number in torchlib
    # So this is a no-op
    return op.Identity(self)


@torch_op("aten::view_as_real", complex=True)
def aten_view_as_real(self: TTensor) -> TTensor:
    """view_as_real(Tensor(a) self) -> Tensor(a)"""

    # We always operate on the real representation of a complex number in torchlib
    # So this is a no-op
    return op.Identity(self)


@torch_op("aten::view_as_real_copy", complex=True)
def aten_view_as_real_copy(self: TTensor) -> TTensor:
    """view_as_real_copy(Tensor self) -> Tensor"""

    # We always operate on the real representation of a complex number in torchlib
    # So this is a no-op
    return op.Identity(self)


@torch_op("aten::view_copy")
def aten_view_copy(self: TTensor, size: IntType) -> TTensor:
    """view_copy(Tensor self, SymInt[] size) -> Tensor"""

    size = op.Cast(size, to=INT64.dtype)  # Reshape only support INT64 as second input
    return op.Reshape(self, size)


@torch_op("aten::vstack")
def aten_vstack(tensors: Sequence[TTensor]) -> TTensor:
    """vstack(Tensor[] tensors) -> Tensor"""

    # The same logic as atleast_2d duplicated here to keep
    # the function self contained
    @graph()
    def reshape_to_2d(tensor):
        shape = op.Shape(tensor)
        rank = op.Size(shape)
        if rank <= 1:
            tensor = op.Reshape(tensor, op.Constant(value_ints=[1, -1]))
        return tensor

    tensors_2d = op.SequenceMap(tensors, body=reshape_to_2d)
    return op.ConcatFromSequence(tensors_2d, axis=0)


@torch_op(("aten::where", "aten::where.self"))
def aten_where(condition: BOOL, self: TTensor, other: TTensor) -> TTensor:
    """where.self(Tensor condition, Tensor self, Tensor other) -> Tensor"""

    return op.Where(condition, self, other)


def aten_xor(self: TensorType, other: TensorType) -> TensorType:
    """__xor__.Tensor(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


@torch_op("aten::zeros")
def aten_zeros(size: IntType, dtype: int = FLOAT.dtype):
    """zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""

    size = op.Cast(size, to=INT64.dtype)
    zero = op.Constant(value_float=0.0)
    zero = op.Cast(zero, to=dtype)

    return op.Expand(zero, size)


@torch_op("aten::zeros_like", trace_only=True)
def aten_zeros_like(self: TTensor, dtype: int = -1) -> TTensor:
    """zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""

    # NOTE: trace_only because both if branches need to be the same type, but we have
    # a cast in the if branch.

    if dtype == -1:
        zero = op.CastLike(0, self)
    else:
        zero = op.Cast(0, to=dtype)

    return _aten_zeros_like_onnx(self, zero)


@torch_op("aten::zeros_like", private=True)
def _aten_zeros_like_onnx(self: TTensor, zero) -> TTensor:
    shape = op.Shape(self)
    return op.Expand(zero, shape)
