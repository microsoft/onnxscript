# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import onnx

import onnxscript.ir as ir
import onnxscript.rewriter.pattern as orp

op = orp.onnxop


class TransposeCast1(orp.RewriteRuleAsClass):
    """Replaces ``Cast + Transpose(. perm=[1, 0])`` by ``TransposeCast2D``."""

    @classmethod
    def pattern(cls, op, x, perm, to):
        return op.Cast(op.Transpose(x, perm=perm), to=to)

    @classmethod
    def check(cls, context, x, perm, to) -> bool:
        if isinstance(perm, ir.RefAttr) or isinstance(to, ir.RefAttr):
            return False
        if perm.value != [1, 0]:
            return False
        if to.value not in {onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT}:
            return False
        return True

    @classmethod
    def rewrite(cls, op, x, perm, to):
        if to.value == onnx.TensorProto.FLOAT:
            return op.Transpose2DCastFP32(x, domain="ai.onnx.contrib")
        return op.Transpose2DCastFP16(x, domain="ai.onnx.contrib")


class TransposeCast2(orp.RewriteRuleAsClass):
    """Replaces ``Transpose(. perm=[1, 0]) + Cast`` by ``TransposeCast2D``."""

    @classmethod
    def pattern(cls, op, x, perm, to):
        return op.Transpose(op.Cast(x, to=to), perm=perm)

    @classmethod
    def check(cls, context, x, perm, to) -> bool:
        if isinstance(perm, ir.RefAttr) or isinstance(to, ir.RefAttr):
            return False
        if perm.value != [1, 0]:
            return False
        if to.value not in {onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT}:
            return False
        return True

    @classmethod
    def rewrite(cls, op, x, perm, to):
        if to.value == onnx.TensorProto.FLOAT:
            return op.Transpose2DCastFP32(x, domain="ai.onnx.contrib")
        return op.Transpose2DCastFP16(x, domain="ai.onnx.contrib")


def llm_rule_set() -> orp.RewriteRuleSet:
    """Returns a set of rules fusing nodes into custom kernels.

    Returns:
        RewriteRuleSet
    """
    return orp.RewriteRuleSet(
        [
            orp.make_rewrite_rule_from_class(TransposeCast1),
            orp.make_rewrite_rule_from_class(TransposeCast2),
        ]
    )
