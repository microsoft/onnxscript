# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

import onnx_ir as ir

from onnxscript.rewriter._rewrite_rule import RewriteRule, RewriteRuleSet

logger = logging.getLogger(__name__)


def cast_constant_of_shape(op, shape, scalar, dtype):
    constant = op.ConstantOfShape(shape, value=scalar)
    return op.Cast(constant, to=dtype)


def fused_cast_constant_of_shape(op, shape: ir.Value, scalar: ir.Attr, dtype: ir.Attr, **_):
    # Cast scalar (a TensorProto attribute) to the specified dtype
    scalar_value = scalar.as_tensor().numpy().item()
    cast_value = ir.tensor([scalar_value], dtype=ir.DataType(dtype.as_int()))
    return op.ConstantOfShape(shape, value=cast_value)


def check_cast_constant_of_shape(context, shape, scalar, dtype, **_) -> bool:
    del context, shape
    if dtype.is_ref() or scalar.is_ref():
        return False
    if dtype.type != ir.AttributeType.INT:
        return False
    to_dtype = dtype.as_int()
    if to_dtype in (ir.DataType.UNDEFINED, ir.DataType.STRING):
        return False
    if scalar.type != ir.AttributeType.TENSOR or scalar.value is None:
        return False
    scalar_tensor = scalar.as_tensor()
    if scalar_tensor.size != 1 or scalar_tensor.dtype == ir.DataType.STRING:
        return False
    return True


def cast_constant_of_shape_without_value(op, shape, dtype):
    constant = op.ConstantOfShape(shape, _allow_other_attributes=False)
    return op.Cast(constant, to=dtype)


def fused_cast_constant_of_shape_without_value(op, shape, dtype, **_):
    zero = ir.tensor([0], dtype=ir.DataType(dtype.as_int()))
    return op.ConstantOfShape(shape, value=zero)


def check_cast_constant_of_shape_without_value(context, shape, dtype, **_) -> bool:
    del context, shape
    if dtype.is_ref():
        return False
    if dtype.type != ir.AttributeType.INT:
        return False
    to_dtype = dtype.as_int()
    if to_dtype in (ir.DataType.UNDEFINED, ir.DataType.STRING):
        return False
    return True


cast_constant_of_shape_rule = RewriteRule(
    cast_constant_of_shape,
    fused_cast_constant_of_shape,
    check_cast_constant_of_shape,
)

cast_constant_of_shape_without_value_rule = RewriteRule(
    cast_constant_of_shape_without_value,
    fused_cast_constant_of_shape_without_value,
    check_cast_constant_of_shape_without_value,
)

rules = RewriteRuleSet(
    [
        cast_constant_of_shape_rule,
        cast_constant_of_shape_without_value_rule,
    ]
)
