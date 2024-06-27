# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

import onnx.helper

from onnxscript import ir
from onnxscript.rewriter import pattern

op = pattern.onnxop
logger = logging.getLogger(__name__)


def cast_constant_of_shape(op, shape, scalar, dtype):
    constant = op.ConstantOfShape(shape, value=scalar)
    return op.Cast(constant, to=dtype)


def fused_cast_constant_of_shape(op, shape: ir.Value, scalar: ir.Attr, dtype: ir.Attr, **_):
    # Cast scalar (a TensorProto attribute) to the specified dtype
    scalar_value = scalar.value.numpy().item()
    cast_value = onnx.helper.make_tensor("value", dtype.value, (1,), [scalar_value])
    return op.ConstantOfShape(shape, value=cast_value)


def cast_constant_of_shape_without_value(op, shape, dtype):
    constant = op.ConstantOfShape(shape)
    return op.Cast(constant, to=dtype)


def fused_cast_constant_of_shape_without_value(op, shape, dtype, **_):
    zero = onnx.helper.make_tensor("value", dtype.value, (1,), [0])
    return op.ConstantOfShape(shape, value=zero)


cast_constant_of_shape_rule = pattern.RewriteRule(
    cast_constant_of_shape, fused_cast_constant_of_shape
)

cast_constant_of_shape_without_value_rule = pattern.RewriteRule(
    cast_constant_of_shape_without_value, fused_cast_constant_of_shape_without_value
)

rules = pattern.RewriteRuleSet(
    [
        cast_constant_of_shape_rule,
        cast_constant_of_shape_without_value_rule,
    ]
)
