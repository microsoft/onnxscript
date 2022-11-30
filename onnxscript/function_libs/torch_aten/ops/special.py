# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""torch.ops.aten operators under the `special` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""
from __future__ import annotations

from typing import Optional, Sequence

from onnxscript import TensorType


def aten_special_airy_ai(x: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_bessel_j0(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_bessel_j1(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_bessel_y0(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_bessel_y1(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_chebyshev_polynomial_t(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_chebyshev_polynomial_u(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_chebyshev_polynomial_v(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_chebyshev_polynomial_w(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_digamma(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_entr(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_erf(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_erfc(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_erfcx(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_erfinv(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_exp2(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_expit(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_expm1(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_gammainc(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_gammaincc(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_gammaln(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_hermite_polynomial_h(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_hermite_polynomial_he(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_i0(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_i0e(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_i1(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_i1e(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_laguerre_polynomial_l(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_legendre_polynomial_p(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_log1p(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_log_ndtr(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_log_softmax(
    self: TensorType, dim: int, *, dtype: Optional[int] = None
) -> TensorType:
    raise NotImplementedError()


def aten_special_logit(self: TensorType, eps: Optional[float] = None) -> TensorType:
    raise NotImplementedError()


def aten_special_logsumexp(
    self: TensorType, dim: Sequence[int], keepdim: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_special_modified_bessel_i0(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_modified_bessel_i1(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_modified_bessel_k0(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_modified_bessel_k1(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_multigammaln(self: TensorType, p: int) -> TensorType:
    raise NotImplementedError()


def aten_special_ndtr(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_ndtri(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_polygamma(n: int, self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_psi(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_round(self: TensorType, *, decimals: int = 0) -> TensorType:
    raise NotImplementedError()


def aten_special_scaled_modified_bessel_k0(x: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_scaled_modified_bessel_k1(x: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_shifted_chebyshev_polynomial_t(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_shifted_chebyshev_polynomial_u(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_shifted_chebyshev_polynomial_v(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_shifted_chebyshev_polynomial_w(x: TensorType, n: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_sinc(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_softmax(
    self: TensorType, dim: int, dtype: Optional[int] = None
) -> TensorType:
    raise NotImplementedError()


def aten_special_spherical_bessel_j0(x: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_xlog1py(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_xlogy(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_special_zeta(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()
