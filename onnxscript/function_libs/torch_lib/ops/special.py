# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
"""torch.ops.aten operators under the `special` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from onnxscript import ir
from onnxscript.function_libs.torch_lib.ops import common as common_ops
from onnxscript.function_libs.torch_lib.registration import torch_op
from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType

_MATH_PI = math.pi

# Coefficients adapted from SciPy XSF's Cephes ``ndtr.h`` (pinned source revision
# 5dbdff8de0dab99b475076612ea227d3f29d6cf6). The original Cephes Math Library
# Release 2.2 is copyright Stephen L. Moshier (1984, 1987, 1988, 1992); SciPy's
# C++ translation is BSD-3-Clause. See THIRD_PARTY_NOTICES.md for the license.
_ERFCX_P = (
    2.46196981473530512524e-10,
    5.64189564831068821977e-1,
    7.46321056442269912687,
    4.86371970985681366614e1,
    1.96520832956077098242e2,
    5.26445194995477358631e2,
    9.34528527171957607540e2,
    1.02755188689515710272e3,
    5.57535335369399327526e2,
)
_ERFCX_Q = (
    1.0,
    1.32281951154744992508e1,
    8.67072140885989742329e1,
    3.54937778887819891062e2,
    9.75708501743205489753e2,
    1.82390916687909736289e3,
    2.24633760818710981792e3,
    1.65666309194161350182e3,
    5.57535340817727675546e2,
)
_ERFCX_R = (
    5.64189583547755073984e-1,
    1.27536670759978104416,
    5.01905042251180477414,
    6.16021097993053585195,
    7.40974269950448939160,
    2.97886665372100240670,
)
_ERFCX_S = (
    1.0,
    2.26052863220117276590,
    9.39603524938001434673,
    1.20489539808096656605e1,
    1.70814450747565897222e1,
    9.60896809063285878198,
    3.36907645100081516050,
)
_ERFCX_T = (
    9.60497373987051638749,
    9.00260197203842689217e1,
    2.23200534594684319226e3,
    7.00332514112805075473e3,
    5.55923013010394962768e4,
)
_ERFCX_U = (
    1.0,
    3.35617141647503099647e1,
    5.21357949780152679795e2,
    4.59432382970980127987e3,
    2.26290000613890934246e4,
    4.92673942608635921086e4,
)


def _erfcx_constant(value: float, like: TFloat) -> TFloat:
    """Creates a coefficient with float64 source precision and the input dtype."""
    return op.CastLike(common_ops.constant(value, dtype=ir.DataType.DOUBLE), like)


def _erfcx_polynomial(coefficients: Sequence[float], x: TFloat) -> TFloat:
    """Emits Horner evaluation; the fixed loop is unrolled while tracing."""
    result = _erfcx_constant(coefficients[0], x)
    for coefficient in coefficients[1:]:
        result = result * x + _erfcx_constant(coefficient, x)
    return result


def aten_special_airy_ai(x: TensorType) -> TensorType:
    """special_airy_ai(Tensor x) -> Tensor"""

    raise NotImplementedError()


def aten_special_bessel_j0(self: TensorType) -> TensorType:
    """special_bessel_j0(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_bessel_j1(self: TensorType) -> TensorType:
    """special_bessel_j1(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_bessel_y0(self: TensorType) -> TensorType:
    """special_bessel_y0(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_bessel_y1(self: TensorType) -> TensorType:
    """special_bessel_y1(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_chebyshev_polynomial_t(x: TensorType, n: TensorType) -> TensorType:
    """special_chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


def aten_special_chebyshev_polynomial_u(x: TensorType, n: TensorType) -> TensorType:
    """special_chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


def aten_special_chebyshev_polynomial_v(x: TensorType, n: TensorType) -> TensorType:
    """special_chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


def aten_special_chebyshev_polynomial_w(x: TensorType, n: TensorType) -> TensorType:
    """special_chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


def aten_special_digamma(self: TensorType) -> TensorType:
    """special_digamma(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_entr(self: TensorType) -> TensorType:
    """special_entr(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::erf", "aten::special_erf"))
def aten_special_erf(self: TFloat) -> TFloat:
    """erf(Tensor self) -> Tensor"""

    return op.Erf(self)


@torch_op(("aten::erfc", "aten::special_erfc"))
def aten_special_erfc(self: TFloat) -> TFloat:
    """erfc(Tensor self) -> Tensor"""

    return op.Sub(1, op.Erf(self))


def _aten_special_erfcx(self: TFloat) -> TFloat:
    """special_erfcx(Tensor self) -> Tensor"""

    # erfcx(x) is evaluated as a positive function of |x|, then reflected for
    # negative x. Bound each rational approximation's input before evaluating it
    # because ONNX Where evaluates both branches.
    abs_self = op.Abs(self)
    zero = _erfcx_constant(0.0, self)
    one = _erfcx_constant(1.0, self)
    two = _erfcx_constant(2.0, self)
    eight = _erfcx_constant(8.0, self)

    central_x = op.Where(op.Less(abs_self, one), abs_self, zero)
    central_z = central_x * central_x
    central_p = _erfcx_polynomial(_ERFCX_T, central_z)
    central_q = _erfcx_polynomial(_ERFCX_U, central_z)
    central_erf = central_x * central_p / central_q
    central = op.Exp(central_z) * (one - central_erf)

    middle_mask = op.And(op.GreaterOrEqual(abs_self, one), op.Less(abs_self, eight))
    middle_x = op.Where(middle_mask, abs_self, one)
    middle_p = _erfcx_polynomial(_ERFCX_P, middle_x)
    middle_q = _erfcx_polynomial(_ERFCX_Q, middle_x)
    middle = middle_p / middle_q

    tail_x = op.Where(op.GreaterOrEqual(abs_self, eight), abs_self, eight)
    tail_r = op.Div(one, tail_x)
    # The tail's denominator has one higher degree than its numerator. Reversing
    # the polynomials in 1 / |x| avoids overflow for large finite inputs.
    tail_p = _erfcx_polynomial(_ERFCX_R[::-1], tail_r)
    tail_q = _erfcx_polynomial(_ERFCX_S[::-1], tail_r)
    tail = tail_r * tail_p / tail_q

    positive = op.Where(
        op.Less(abs_self, one), central, op.Where(op.Less(abs_self, eight), middle, tail)
    )
    reflected = op.Sub(op.Mul(two, op.Exp(op.Mul(self, self))), positive)
    result = op.Where(op.Less(self, zero), reflected, positive)
    return op.Where(op.IsNaN(self), self, result)


@torch_op("aten::special_erfcx", trace_only=True)
def aten_special_erfcx(self: TFloat) -> TFloat:
    """special_erfcx(Tensor self) -> Tensor"""

    # The degree-eight middle polynomial overflows float16 even though the
    # final ratio is finite. Evaluate low-precision inputs in float32, then
    # restore the requested dtype. Float32 and float64 retain their precision.
    if self.dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16):
        return op.CastLike(_aten_special_erfcx(op.Cast(self, to=ir.DataType.FLOAT)), self)
    return _aten_special_erfcx(self)


def aten_special_erfinv(self: TensorType) -> TensorType:
    """special_erfinv(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_exp2(self: TensorType) -> TensorType:
    """special_exp2(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_expit(self: TensorType) -> TensorType:
    """special_expit(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::expm1", "aten::special_expm1"))
def aten_special_expm1(self: TFloat) -> TFloat:
    """special_expm1(Tensor self) -> Tensor"""

    return op.Sub(op.Exp(self), 1)


def aten_special_gammainc(self: TensorType, other: TensorType) -> TensorType:
    """special_gammainc(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_special_gammaincc(self: TensorType, other: TensorType) -> TensorType:
    """special_gammaincc(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


def aten_special_gammaln(self: TensorType) -> TensorType:
    """special_gammaln(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_hermite_polynomial_h(x: TensorType, n: TensorType) -> TensorType:
    """special_hermite_polynomial_h(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


def aten_special_hermite_polynomial_he(x: TensorType, n: TensorType) -> TensorType:
    """special_hermite_polynomial_he(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


def aten_special_i0(self: TensorType) -> TensorType:
    """special_i0(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_i0e(self: TensorType) -> TensorType:
    """special_i0e(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_i1(self: TensorType) -> TensorType:
    """special_i1(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_i1e(self: TensorType) -> TensorType:
    """special_i1e(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_laguerre_polynomial_l(x: TensorType, n: TensorType) -> TensorType:
    """special_laguerre_polynomial_l(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


def aten_special_legendre_polynomial_p(x: TensorType, n: TensorType) -> TensorType:
    """special_legendre_polynomial_p(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


def aten_special_log1p(self: TensorType) -> TensorType:
    """special_log1p(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_log_ndtr(self: TensorType) -> TensorType:
    """special_log_ndtr(Tensor self) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::log_softmax.int", "aten::special_log_softmax"), trace_only=True)
def aten_special_log_softmax(self: TFloat, dim: int, dtype: int = -1) -> TFloat:
    """special_log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""

    self_is_scalar = len(self.shape) == 0
    if self_is_scalar:
        self = op.Unsqueeze(self, [0])
    result = op.LogSoftmax(self, axis=dim)
    if dtype != -1:
        result = op.Cast(result, to=dtype)
    if self_is_scalar:  # squeeze to scalar due to input is scalar
        result = op.Squeeze(result)
    return result


def aten_special_logit(self: TensorType, eps: Optional[float] = None) -> TensorType:
    """special_logit(Tensor self, float? eps=None) -> Tensor"""
    # TODO: alias of core.aten_logit
    raise NotImplementedError()


def aten_special_logsumexp(
    self: TensorType, dim: Sequence[int], keepdim: bool = False
) -> TensorType:
    """special_logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor"""

    raise NotImplementedError()


def aten_special_modified_bessel_i0(self: TensorType) -> TensorType:
    """special_modified_bessel_i0(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_modified_bessel_i1(self: TensorType) -> TensorType:
    """special_modified_bessel_i1(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_modified_bessel_k0(self: TensorType) -> TensorType:
    """special_modified_bessel_k0(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_modified_bessel_k1(self: TensorType) -> TensorType:
    """special_modified_bessel_k1(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_multigammaln(self: TensorType, p: int) -> TensorType:
    """special_multigammaln(Tensor self, int p) -> Tensor"""

    raise NotImplementedError()


def aten_special_ndtr(self: TensorType) -> TensorType:
    """special_ndtr(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_ndtri(self: TensorType) -> TensorType:
    """special_ndtri(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_polygamma(n: int, self: TensorType) -> TensorType:
    """special_polygamma(int n, Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_psi(self: TensorType) -> TensorType:
    """special_psi(Tensor self) -> Tensor"""

    raise NotImplementedError()


def aten_special_round(self: TensorType, decimals: int = 0) -> TensorType:
    """special_round(Tensor self, *, int decimals=0) -> Tensor"""

    raise NotImplementedError()


def aten_special_scaled_modified_bessel_k0(x: TensorType) -> TensorType:
    """special_scaled_modified_bessel_k0(Tensor x) -> Tensor"""

    raise NotImplementedError()


def aten_special_scaled_modified_bessel_k1(x: TensorType) -> TensorType:
    """special_scaled_modified_bessel_k1(Tensor x) -> Tensor"""

    raise NotImplementedError()


def aten_special_shifted_chebyshev_polynomial_t(x: TensorType, n: TensorType) -> TensorType:
    """special_shifted_chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


def aten_special_shifted_chebyshev_polynomial_u(x: TensorType, n: TensorType) -> TensorType:
    """special_shifted_chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


def aten_special_shifted_chebyshev_polynomial_v(x: TensorType, n: TensorType) -> TensorType:
    """special_shifted_chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


def aten_special_shifted_chebyshev_polynomial_w(x: TensorType, n: TensorType) -> TensorType:
    """special_shifted_chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::special_sinc", "aten::sinc"))
def aten_special_sinc(self: TFloat) -> TFloat:
    """special_sinc(Tensor self) -> Tensor"""

    # This computes the normalized sinc, where the input is multiplied by pi.
    # https://pytorch.org/docs/stable/special.html#torch.special.sinc
    pi_self = self * _MATH_PI

    return op.Where(self == 0.0, op.CastLike(1, self), op.Sin(pi_self) / pi_self)


def aten_special_spherical_bessel_j0(x: TensorType) -> TensorType:
    """special_spherical_bessel_j0(Tensor x) -> Tensor"""

    raise NotImplementedError()


def aten_special_xlog1py(self: TensorType, other: TensorType) -> TensorType:
    """special_xlog1py(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()


@torch_op(("aten::xlogy.Tensor", "aten::xlogy.Scalar_Self", "aten::xlogy.Scalar_Other"))
def aten_special_xlogy(self: TFloat, other: TFloat) -> TFloat:
    """special_xlogy(Tensor self, Tensor other) -> Tensor"""

    # https://pytorch.org/docs/stable/special.html#torch.special.xlogy
    # out := {
    #     NaN if other == NaN
    #     0 if self == 0
    #     self * log(other) otherwise
    # }

    nans = op.IsNaN(other)
    zeros = op.Equal(self, 0)
    xlogy = op.Mul(self, op.Log(other))
    xlogy_with_nans = op.Where(nans, other, xlogy)
    return op.Where(zeros, self, xlogy_with_nans)


def aten_special_zeta(self: TensorType, other: TensorType) -> TensorType:
    """special_zeta(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError()
