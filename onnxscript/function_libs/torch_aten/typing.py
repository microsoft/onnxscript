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

FloatType = Union[FLOAT16, FLOAT, DOUBLE]
IntType = Union[INT8, INT16, INT32, INT64]
UnlessBool = Union[
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
    COMPLEX64,
    COMPLEX128,
]

TFloat = TypeVar("TFloat", bound=FloatType)
TInt = TypeVar("TInt", bound=IntType)
TBool = TypeVar("TBool", bound=BOOL)
TUnlessBool = TypeVar("TUnlessBool", bound=UnlessBool)
