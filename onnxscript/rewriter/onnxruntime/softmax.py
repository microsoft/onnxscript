# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import logging

import onnx

from onnxscript import ir
from onnxscript.rewriter import pattern

logger = logging.getLogger(__name__)


def softmax_with_fp32_upcast(op, input, axis):
    upcast = op.Cast(input, to=onnx.TensorProto.FLOAT)
    softmax = op.Softmax(upcast, axis=axis)  # pylint: disable=redefined-outer-name
    return op.Cast(softmax, to=onnx.TensorProto.FLOAT16)


def softmax(op, input, axis):
    return op.Softmax(input, axis=axis)


def softmax_with_fp32_upcast_without_axis(op, input):
    upcast = op.Cast(input, to=onnx.TensorProto.FLOAT)
    softmax = op.Softmax(upcast)  # pylint: disable=redefined-outer-name
    return op.Cast(softmax, to=onnx.TensorProto.FLOAT16)


def softmax_without_axis(op, input):
    return op.Softmax(input)


def check_if_fp16_input(context, input, **_) -> bool:
    if input is None:
        logger.warning(
            "Cannot perform softmax upcast removal: "
            "cannot retrieve match_bindings for 'input' for dtype validation."
        )
        return False
    return input.dtype == ir.DataType.FLOAT16


# pylint: disable=pointless-string-statement
"""
This is an onnxruntime specific pattern. Softmax upcast is a common
pattern observed in transformers models to prevent overflow. However
this is not required since onnxruntime implementation already takes
overflow into account. Hence it is safe to remove the surrounding casts
to free up memory as well as saving performance.
"""
# pylint: enable=pointless-string-statement
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
