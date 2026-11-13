# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

import onnx_ir as ir

from onnxscript.rewriter._basics import MatchFailureError
from onnxscript.rewriter._rewrite_rule import RewriteRule, RewriteRuleSet

logger = logging.getLogger(__name__)


def _require_concrete_attributes(*attributes: ir.Attr) -> None:
    for attribute in attributes:
        if attribute.is_ref():
            raise MatchFailureError(
                f"Cannot fold attribute {attribute.name!r}: it references function "
                f"parameter {attribute.ref_attr_name!r}, whose value is unknown."
            )


def cast_constant_of_shape(op, shape, scalar, dtype):
    constant = op.ConstantOfShape(shape, value=scalar)
    return op.Cast(constant, to=dtype)


def fused_cast_constant_of_shape(op, shape: ir.Value, scalar: ir.Attr, dtype: ir.Attr, **_):
    _require_concrete_attributes(scalar, dtype)
    # Cast scalar (a TensorProto attribute) to the specified dtype
    scalar_value = scalar.value.numpy().item()
    cast_value = ir.tensor([scalar_value], dtype=ir.DataType(dtype.as_int()))
    return op.ConstantOfShape(shape, value=cast_value)


def check_cast_constant_of_shape(context, shape, scalar, dtype, **_) -> bool:
    _require_concrete_attributes(scalar, dtype)
    return True


def cast_constant_of_shape_without_value(op, shape, dtype):
    constant = op.ConstantOfShape(shape, _allow_other_attributes=False)
    return op.Cast(constant, to=dtype)


def fused_cast_constant_of_shape_without_value(op, shape, dtype, **_):
    _require_concrete_attributes(dtype)
    zero = ir.tensor([0], dtype=ir.DataType(dtype.as_int()))
    return op.ConstantOfShape(shape, value=zero)


def check_cast_constant_of_shape_without_value(context, shape, dtype, **_) -> bool:
    _require_concrete_attributes(dtype)
    return True


cast_constant_of_shape_rule = RewriteRule(
    cast_constant_of_shape, fused_cast_constant_of_shape, check_cast_constant_of_shape
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
