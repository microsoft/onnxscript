# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""torch.ops.aten operators under the `linalg` module.

- No inplace operators.
- All functions should not have the script() decorator. This is because
    we want to delay the compilation of the function.
"""
from __future__ import annotations

from typing import Optional, Sequence

from onnxscript import TensorType


def aten_linalg_cholesky(self: TensorType, *, upper: bool = False) -> TensorType:
    raise NotImplementedError()


def aten_linalg_cholesky_ex(
    self: TensorType, *, upper: bool = False, check_errors: bool = False
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_cond(self: TensorType, p: Optional[float] = None) -> TensorType:
    raise NotImplementedError()


def aten_linalg_cross(self: TensorType, other: TensorType, *, dim: int = -1) -> TensorType:
    raise NotImplementedError()


def aten_linalg_det(A: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_linalg_diagonal(
    A: TensorType, *, offset: int = 0, dim1: int = -2, dim2: int = -1
) -> TensorType:
    raise NotImplementedError()


def aten_linalg_eig(self: TensorType) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_eigh(self: TensorType, UPLO: str = "L") -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_eigvals(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_linalg_eigvalsh(self: TensorType, UPLO: str = "L") -> TensorType:
    raise NotImplementedError()


def aten_linalg_householder_product(input: TensorType, tau: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_linalg_inv(A: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_linalg_inv_ex(
    A: TensorType, *, check_errors: bool = False
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_ldl_factor(
    self: TensorType, *, hermitian: bool = False
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_ldl_factor_ex(
    self: TensorType, *, hermitian: bool = False, check_errors: bool = False
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_ldl_solve(
    LD: TensorType, pivots: TensorType, B: TensorType, *, hermitian: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_linalg_lstsq(
    self: TensorType,
    b: TensorType,
    rcond: Optional[float] = None,
    *,
    driver: Optional[str] = None,
) -> tuple[TensorType, TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_lu(
    A: TensorType, *, pivot: bool = True
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_lu_factor(
    A: TensorType, *, pivot: bool = True
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_lu_factor_ex(
    A: TensorType, *, pivot: bool = True, check_errors: bool = False
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_lu_solve(
    LU: TensorType,
    pivots: TensorType,
    B: TensorType,
    *,
    left: bool = True,
    adjoint: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_linalg_matmul(self: TensorType, other: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_linalg_matrix_exp(self: TensorType) -> TensorType:
    raise NotImplementedError()


def aten_linalg_matrix_norm(
    self: TensorType,
    ord: float,
    dim: Sequence[int] = (-2, -1),
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_linalg_matrix_power(self: TensorType, n: int) -> TensorType:
    raise NotImplementedError()


def aten_linalg_matrix_rank(
    self: TensorType, tol: float, hermitian: bool = False
) -> TensorType:
    raise NotImplementedError()


def aten_linalg_multi_dot(tensors: TensorType[...]) -> TensorType:
    raise NotImplementedError()


def aten_linalg_norm(
    self: TensorType,
    ord: Optional[float] = None,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
) -> TensorType:
    raise NotImplementedError()


def aten_linalg_pinv(self: TensorType, rcond: float, hermitian: bool = False) -> TensorType:
    raise NotImplementedError()


def aten_linalg_qr(A: TensorType, mode: str = "reduced") -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_slogdet(A: TensorType) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_solve(A: TensorType, B: TensorType, *, left: bool = True) -> TensorType:
    raise NotImplementedError()


def aten_linalg_solve_ex(
    A: TensorType, B: TensorType, *, left: bool = True, check_errors: bool = False
) -> tuple[TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_solve_triangular(
    self: TensorType,
    B: TensorType,
    *,
    upper: bool = None,
    left: bool = True,
    unitriangular: bool = False,
) -> TensorType:
    raise NotImplementedError()


def aten_linalg_svd(
    A: TensorType, full_matrices: bool = True, *, driver: Optional[str] = None
) -> tuple[TensorType, TensorType, TensorType]:
    raise NotImplementedError()


def aten_linalg_svdvals(A: TensorType, *, driver: Optional[str] = None) -> TensorType:
    raise NotImplementedError()


def aten_linalg_tensorinv(self: TensorType, ind: int = 2) -> TensorType:
    raise NotImplementedError()


def aten_linalg_tensorsolve(
    self: TensorType, other: TensorType, dims: Optional[int] = None
) -> TensorType:
    raise NotImplementedError()


def aten_linalg_vander(x: TensorType, *, N: Optional[int] = None) -> TensorType:
    raise NotImplementedError()


def aten_linalg_vecdot(x: TensorType, y: TensorType, *, dim: int = -1) -> TensorType:
    raise NotImplementedError()


def aten_linalg_vector_norm(
    self: TensorType,
    ord: int = 2,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[int] = None,
) -> TensorType:
    raise NotImplementedError()
