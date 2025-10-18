# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Remove optional bias when it is all zero from Conv, ConvTranspose, Gemm and QLinearConv operations."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from onnxscript import ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class _RemoveOptionalBias(RewriteRuleClassBase):
    def rewrite(self, op: ir.tape.Tape, out: ir.Value, **_) -> ir.Value:
        node = out.producer()

        return op.op(
            self.op_type,
            inputs=node.inputs[:-1],
            attributes=node.attributes,
        )

    def check(self, context, b: ir.Value, **_) -> MatchResult:
        """Condition to check if we need to replace the pattern.

        The pattern is applied only when the bias is all zeros. The bias should be
        a constant value (i.e., provided by Constant nodes or initializers).

        Returns:
            MatchResult:
                Success if we need to replace the pattern, Failure otherwise.
        """
        del context  # Unused
        check_result = MatchResult()

        # Check if bias is a constant/initializer
        bias_tensor = ir.convenience.get_const_tensor(b)
        if bias_tensor is None:
            return check_result.fail("Bias is not a constant/initializer.")

        # Check if bias is all zeros
        bias_array = bias_tensor.numpy()
        if not np.equal(bias_array, 0.0).all():
            return check_result.fail("Bias is not all zeros.")

        return check_result


class RemoveOptionalBiasFromConv(_RemoveOptionalBias):
    """Remove zero bias from Conv operation."""

    op_type: ClassVar[str] = "Conv"

    def pattern(self, op: ir.tape.Tape, x: ir.Value, w: ir.Value, b: ir.Value) -> ir.Value:
        return op.Conv(x, w, b, _outputs=["out"])


class RemoveOptionalBiasFromConvTranspose(_RemoveOptionalBias):
    """Remove zero bias from ConvTranspose operation."""

    op_type: ClassVar[str] = "ConvTranspose"

    def pattern(self, op: ir.tape.Tape, x: ir.Value, w: ir.Value, b: ir.Value) -> ir.Value:
        return op.ConvTranspose(x, w, b, _outputs=["out"])


class RemoveOptionalBiasFromQLinearConv(_RemoveOptionalBias):
    """Remove zero bias from QLinearConv operation."""

    op_type: ClassVar[str] = "QLinearConv"

    def pattern(
        self,
        op: ir.tape.Tape,
        x,
        x_scale,
        x_zero_point,
        w,
        w_scale,
        w_zero_point,
        y_scale,
        y_zero_point,
        b: ir.Value,
    ) -> ir.Value:
        return op.QLinearConv(
            x,
            x_scale,
            x_zero_point,
            w,
            w_scale,
            w_zero_point,
            y_scale,
            y_zero_point,
            b,
            _outputs=["out"],
        )


class RemoveOptionalBiasFromGemm(_RemoveOptionalBias):
    """Remove zero bias from Gemm operation."""

    op_type: ClassVar[str] = "Gemm"

    def pattern(self, op: ir.tape.Tape, x: ir.Value, w: ir.Value, b: ir.Value) -> ir.Value:
        return op.Gemm(x, w, b, _outputs=["out"])


remove_optional_bias_from_conv_rule = RemoveOptionalBiasFromConv().rule()
remove_optional_bias_from_conv_transpose_rule = RemoveOptionalBiasFromConvTranspose().rule()
remove_optional_bias_from_qlinear_conv_rule = RemoveOptionalBiasFromQLinearConv().rule()
remove_optional_bias_from_gemm_rule = RemoveOptionalBiasFromGemm().rule()

rules = RewriteRuleSet(
    [
        remove_optional_bias_from_conv_rule,
        remove_optional_bias_from_conv_transpose_rule,
        remove_optional_bias_from_qlinear_conv_rule,
        remove_optional_bias_from_gemm_rule,
    ]
)
