# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

import numpy as np
import onnx
import onnx_ir as ir

from onnxscript.rewriter._rewrite_rule import RewriteRule, RewriteRuleSet

logger = logging.getLogger(__name__)


def cast_constant_of_shape(op, shape, scalar, dtype):
    constant = op.ConstantOfShape(shape, value=scalar)
    return op.Cast(constant, to=dtype)


def fused_cast_constant_of_shape(op, shape: ir.Value, scalar: ir.Attr, dtype: ir.Attr, **_):
    # Cast scalar (a TensorProto attribute) to the specified dtype. Convert the array, not a
    # Python scalar, so that values outside the target range wrap like the Cast op instead of
    # raising OverflowError.
    to_dtype = ir.DataType(dtype.as_int())
    with np.errstate(invalid="ignore"):
        cast_array = scalar.value.numpy().astype(to_dtype.numpy()).reshape(1)
    cast_value = ir.tensor(cast_array, dtype=to_dtype)
    return op.ConstantOfShape(shape, value=cast_value)


def _is_supported_constant_of_shape_dtype(context, dtype: ir.Attr) -> bool:
    if dtype.is_ref() or dtype.type != ir.AttributeType.INT:
        return False

    try:
        dtype_name = onnx.TensorProto.DataType.Name(dtype.as_int()).lower()
    except ValueError:
        return False

    opset_imports = context.graph_or_function.opset_imports
    opset = opset_imports.get("")
    if opset is None:
        opset = opset_imports.get("ai.onnx")
    if opset is None:
        return False

    try:
        schema = onnx.defs.get_schema("ConstantOfShape", opset, "")
    except onnx.defs.SchemaError:
        return False

    output_type = schema.outputs[0].type_str
    output_constraint = next(
        (
            constraint
            for constraint in schema.type_constraints
            if constraint.type_param_str == output_type
        ),
        None,
    )
    if output_constraint is None:
        return False
    return f"tensor({dtype_name})" in output_constraint.allowed_type_strs


def check_cast_constant_of_shape(context, shape, scalar, dtype, **_) -> bool:
    del shape, scalar
    return _is_supported_constant_of_shape_dtype(context, dtype)


def cast_constant_of_shape_without_value(op, shape, dtype):
    constant = op.ConstantOfShape(shape)
    return op.Cast(constant, to=dtype)


def fused_cast_constant_of_shape_without_value(op, shape, dtype, **_):
    zero = ir.tensor([0], dtype=ir.DataType(dtype.as_int()))
    return op.ConstantOfShape(shape, value=zero)


def check_cast_constant_of_shape_without_value(context, shape, dtype, **_) -> bool:
    del shape
    return _is_supported_constant_of_shape_dtype(context, dtype)


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
