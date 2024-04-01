from __future__ import annotations

import logging
from typing import Any

import onnx

from onnxscript import ir
from onnxscript.rewriter import pattern

op = pattern.onnxop
logger = logging.getLogger(__name__)


def softmax_with_fp32_upcast(input, axis):
    upcast = op.Cast(input, to=onnx.TensorProto.FLOAT)
    softmax = op.Softmax(upcast, axis=axis)
    return op.Cast(softmax, to=onnx.TensorProto.FLOAT16)


def softmax(input, axis):
    return op.Softmax(input, axis=axis)


def softmax_with_fp32_upcast_without_axis(input):
    upcast = op.Cast(input, to=onnx.TensorProto.FLOAT)
    softmax = op.Softmax(upcast)
    return op.Cast(softmax, to=onnx.TensorProto.FLOAT16)


def softmax_without_axis(input):
    return op.Softmax(input)


def check_if_fp16_input(match_bindings: dict[str, ir.Value | Any]) -> bool:
    input_val = match_bindings.get("input")
    if input_val is None:
        logger.warning(
            "Cannot perform softmax upcast removal: "
            "cannot retrieve match_bindings for 'input' for dtype validation."
        )
        return False
    return input_val.element_type == onnx.TensorProto.FLOAT16


"""
This is an onnxruntime specific pattern. Softmax upcast is a common
pattern observed in transformers models to prevent overflow. However
this is not required since onnxruntime implementation already takes
overflow into account. Hence it is safe to remove the surrounding casts
to free up memory as well as saving performance.
"""
rules = pattern.RewriteRuleSet(
    [
        pattern.RewriteRule(softmax_with_fp32_upcast, softmax, check_if_fp16_input),
        pattern.RewriteRule(
            softmax_with_fp32_upcast_without_axis,
            softmax_without_axis,
            check_if_fp16_input,
        ),
    ]
)
