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
    STRING,
    UINT8,
)

# NOTE: We do not care about unsigned types beyond UINT8 because PyTorch does not us them.
# More detail can be found: https://pytorch.org/docs/stable/tensors.html

TensorType = Union[
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
]
FloatType = Union[FLOAT16, FLOAT, DOUBLE]
IntType = Union[INT8, INT16, INT32, INT64]
RealType = Union[
    BFLOAT16,
    FLOAT16,
    FLOAT,
    DOUBLE,
    INT8,
    INT16,
    INT32,
    INT64,
]

TTensor = TypeVar(
    "TTensor",
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
)
TTensorOrString = TypeVar(
    "TTensorOrString",
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
    STRING,
)
TFloat = TypeVar("TFloat", FLOAT16, FLOAT, DOUBLE)
TFloatOrBFloat16 = TypeVar("TFloatOrBFloat16", FLOAT16, FLOAT, DOUBLE, BFLOAT16)
TFloatOrUInt8 = TypeVar("TFloatOrUInt8", FLOAT, FLOAT16, DOUBLE, INT8, UINT8)
TInt = TypeVar("TInt", INT8, INT16, INT32, INT64)
TReal = TypeVar(
    "TReal",
    BFLOAT16,
    FLOAT16,
    FLOAT,
    DOUBLE,
    INT8,
    INT16,
    INT32,
    INT64,
)
TRealUnlessInt16OrInt8 = TypeVar(
    "TRealUnlessInt16OrInt8", FLOAT16, FLOAT, DOUBLE, BFLOAT16, INT32, INT64
)
TRealUnlessFloat16OrInt8 = TypeVar(
    "TRealUnlessFloat16OrInt8", DOUBLE, FLOAT, INT16, INT32, INT64
)
TrealOrUInt8 = TypeVar(
    "TrealOrUInt8", BFLOAT16, FLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8
)
