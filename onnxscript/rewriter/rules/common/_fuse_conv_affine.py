# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Absorbs affine operation into convolution (best effort):
- Conv(Mul(Add(x))) -> Conv (only conv without padding can be fused)
- Add(Mul(Conv)) -> Conv (for all convolutions)
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from onnxscript.rewriter import pattern
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._ir_utils import get_const_value, get_singleton_value


class _ConvAffineFusionBase(pattern.RewriteRuleClassBase):
    def check(
        self,
        context,
        x: ir.Value,
        w: ir.Value,
        b: ir.Value,
        scale: ir.Value,
        offset: ir.Value,
        conv_out: ir.Value,
    ) -> MatchResult:
        check_result = MatchResult()
        if get_const_value(w) is None:
            return check_result.fail("The weight of Conv should be constant")
        if get_const_value(b) is None:
            return check_result.fail("The bias of Conv should be constant")
        if get_singleton_value(scale) is None:
            return check_result.fail("Operand for Mul should be constant scalar value")
        if get_singleton_value(offset) is None:
            return check_result.fail("Operand for Add should be constant scalar value")
        return check_result


class AffineConvFusion(_ConvAffineFusionBase):
    """Pattern: scalar Mul + scalar Add + Conv (1x1) --> Conv(1x1)"""

    def pattern(
        self, op, x: ir.Value, w: ir.Value, b: ir.Value, scale: ir.Value, offset: ir.Value
    ) -> ir.Value:
        return op.Conv(
            x * scale + offset,
            w,
            b,
            pads=[0, 0, 0, 0],
            _allow_other_attributes=True,
            _outputs=["conv_out"],
        )

    def rewrite(
        self,
        op: ir.tape.Tape,
        x: ir.Value,
        w: ir.Value,
        b: ir.Value,
        scale: ir.Value,
        offset: ir.Value,
        conv_out: ir.Value,
    ) -> ir.Value:
        scale_value = scale.const_value.numpy()  # type: ignore[union-attr]
        offset_value = offset.const_value.numpy()  # type: ignore[union-attr]
        w_value = w.const_value.numpy()  # type: ignore[union-attr]
        b_value = b.const_value.numpy()  # type: ignore[union-attr]
        scaled_w_value = op.initializer(ir.tensor(w_value * scale_value), (w.name or "") + "_scaled")  # type: ignore[operator]
        offset_bias: ir.Tensor | ir.Value = ir.tensor(
            b_value + np.sum(w_value * offset_value, axis=(1, 2, 3), keepdims=False)
        )
        offset_bias = op.initializer(offset_bias, (b.name or "") + "_offset")  # type: ignore[arg-type]
        conv_attributes = conv_out.producer().attributes  # type: ignore[union-attr]
        return op.Conv(x, scaled_w_value, offset_bias, **conv_attributes)


class ConvAffineFusion(_ConvAffineFusionBase):
    """Pattern: Conv + scalar Mul + scalar Add --> Conv(1x1)"""

    def pattern(
        self, op, x: ir.Value, w: ir.Value, b: ir.Value, scale: ir.Value, offset: ir.Value
    ) -> ir.Value:
        return (
            op.Conv(x, w, b, _allow_other_attributes=True, _outputs=["conv_out"]) * scale
            + offset
        )

    def rewrite(
        self,
        op: ir.tape.Tape,
        x: ir.Value,
        w: ir.Value,
        b: ir.Value,
        scale: ir.Value,
        offset: ir.Value,
        conv_out: ir.Value,
    ) -> ir.Value:
        scale_value = scale.const_value.numpy()  # type: ignore[union-attr]
        offset_value = offset.const_value.numpy()  # type: ignore[union-attr]
        w_value = w.const_value.numpy()  # type: ignore[union-attr]
        b_value = b.const_value.numpy()  # type: ignore[union-attr]
        scaled_w_weight = op.initializer(ir.tensor(w_value * scale_value), (w.name or "") + "_scaled")  # type: ignore[operator]
        offset_bias: ir.Tensor | ir.Value = ir.tensor(b_value * scale_value + offset_value)
        offset_bias = op.initializer(offset_bias, (b.name or "") + "_offset")  # type: ignore[arg-type]
        conv_attributes = conv_out.producer().attributes  # type: ignore[union-attr]
        return op.Conv(x, scaled_w_weight, offset_bias, **conv_attributes)


affine_conv_fusion_rule = AffineConvFusion().rule()
conv_affine_fusion_rule = ConvAffineFusion().rule()
