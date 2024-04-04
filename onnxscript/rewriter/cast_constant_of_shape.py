from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import onnx

import onnxscript._legacy_ir as ir
from onnxscript.rewriter import pattern

op = pattern.onnxop
logger = logging.getLogger(__name__)


def cast_constant_of_shape(
    shape: Sequence[int],
    t: Any,
    dtype: int,
    match_bindings: dict[str, ir.Value | Any] | None = None,
) -> pattern.OpPattern:
    constant = op.ConstantOfShape(shape, value=t)
    return op.Cast(constant, to=dtype)


def fused_cast_constant_of_shape(
    shape: Sequence[int], t: Any, dtype: int, match_bindings: dict[str, ir.Value | Any]
) -> pattern.OpPattern:
    del dtype  # unused
    del t  # unused
    v_dtype = match_bindings["dtype"]
    v_t = match_bindings["t"]
    casted_val = onnx.numpy_helper.to_array(v_t).astype(  # type: ignore[arg-type]
        dtype=onnx.helper.tensor_dtype_to_np_dtype(v_dtype)  # type: ignore[arg-type]
    )
    return op.ConstantOfShape(shape, value=casted_val)


def cast_constant_of_shape_without_value(
    shape: Sequence[int],
    dtype: int,
    match_bindings: dict[str, ir.Value | Any] | None = None,
) -> pattern.OpPattern:
    del match_bindings  # Unused
    constant = op.ConstantOfShape(shape)
    return op.Cast(constant, to=dtype)


def fused_cast_constant_of_shape_without_value(
    shape: Sequence[int], dtype: int, match_bindings: dict[str, ir.Value | Any]
) -> pattern.OpPattern:
    del dtype  # Unused
    v_dtype = match_bindings["dtype"]
    val = np.zeros(1, dtype=onnx.helper.tensor_dtype_to_np_dtype(v_dtype))  # type: ignore
    return op.ConstantOfShape(shape, value=val)


cast_constant_of_shape_rule = pattern.RewriteRule(
    cast_constant_of_shape,
    pattern.ReplacementPatternFunction(fused_cast_constant_of_shape, delay_run=True),
)

cast_constant_of_shape_without_value_rule = pattern.RewriteRule(
    cast_constant_of_shape_without_value,
    pattern.ReplacementPatternFunction(
        fused_cast_constant_of_shape_without_value, delay_run=True
    ),
)

rules = pattern.RewriteRuleSet(
    [
        cast_constant_of_shape_rule,
        cast_constant_of_shape_without_value_rule,
    ]
)
