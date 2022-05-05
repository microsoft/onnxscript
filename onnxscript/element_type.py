# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from enum import Enum


class ElementType(Enum):
    """
    ElementType: an enumeration encoding the allowed element types in a tensor
    Corresponds to `TensorProto::DataType`.
    """

    UNDEFINED = 0

    FLOAT = 1   # float
    UINT8 = 2   # uint8_t
    INT8 = 3    # int8_t
    UINT16 = 4  # uint16_t
    INT16 = 5   # int16_t
    INT32 = 6   # int32_t
    INT64 = 7   # int64_t
    STRING = 8  # string
    BOOL = 9    # bool

    # IEEE754 half-precision floating-point format (16 bits wide).
    # This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    FLOAT16 = 10

    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14     # complex with float32 real and imaginary components
    COMPLEX128 = 15    # complex with float64 real and imaginary components

    # Non-IEEE floating-point format based on IEEE754 single-precision
    # floating-point number truncated to 16 bits.
    # This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    BFLOAT16 = 16
