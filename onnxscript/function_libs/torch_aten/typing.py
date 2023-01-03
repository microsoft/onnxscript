# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Typings for function definitions."""

from __future__ import annotations

from typing import TypeVar, Union

from onnxscript import (
    BFLOAT16,
    BOOL,
    COMPLEX64,
    COMPLEX128,
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
)

_TensorType = Union[
    BFLOAT16,
    BOOL,
    COMPLEX64,
    COMPLEX128,
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
]
_FloatType = Union[FLOAT16, FLOAT, DOUBLE]
_IntType = Union[INT8, INT16, INT32, INT64]
_RealType = Union[
    BFLOAT16,
    FLOAT16,
    FLOAT,
    DOUBLE,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
]

TTensor = TypeVar("TTensor", bound=_TensorType)
TFloat = TypeVar("TFloat", bound=_FloatType)
TInt = TypeVar("TInt", bound=_IntType)
TReal = TypeVar("TReal", bound=_RealType)
