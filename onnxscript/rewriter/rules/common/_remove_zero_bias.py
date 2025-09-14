# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Remove optional bias when it is all zero from Conv and related operations."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from onnxscript import ir
from onnxscript.ir import convenience
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class _RemoveZeroBiasBase(RewriteRuleClassBase):
    """Base class for removing zero bias from operations."""

    def rewrite(self, op: ir.tape.Tape, out: ir.Value, **_) -> ir.Value:
        """Remove the bias input from the operation."""
        node = out.producer()

        return op.op(
            self.op_type,
            inputs=node.inputs[:-1],
            attributes=node.attributes,
            domain=node.domain,
        )

    def _check_bias_is_zero(self, bias_value: ir.Value) -> MatchResult:
        """Check if the bias value is present and is all zeros."""
        check_result = MatchResult()

        # Check if bias is a constant/initializer
        bias_tensor = convenience.get_const_tensor(bias_value)
        if bias_tensor is None:
            return check_result.fail("Bias is not a constant/initializer.")

        # Check if bias is all zeros
        bias_array = bias_tensor.numpy()
        if not np.allclose(bias_array, 0.0, atol=1e-8):
            return check_result.fail("Bias is not all zeros.")

        return check_result

    def check(self, context, x: ir.Value, w: ir.Value, b: ir.Value, **_) -> MatchResult:
        """Check if the bias is present and is all zeros."""
        del context  # Unused
        return self._check_bias_is_zero(b)


class RemoveZeroBiasFromConv(_RemoveZeroBiasBase):
    """Remove zero bias from Conv operations."""

    op_type: ClassVar = "Conv"

    def pattern(self, op: ir.tape.Tape, x: ir.Value, w: ir.Value, b: ir.Value) -> ir.Value:
        return op.Conv(x, w, b, _outputs=["out"])


class RemoveZeroBiasFromConvTranspose(_RemoveZeroBiasBase):
    """Remove zero bias from ConvTranspose operations."""

    op_type: ClassVar = "ConvTranspose"

    def pattern(self, op: ir.tape.Tape, x: ir.Value, w: ir.Value, b: ir.Value) -> ir.Value:
        return op.ConvTranspose(x, w, b, _outputs=["out"])


class RemoveZeroBiasFromQLinearConv(_RemoveZeroBiasBase):
    """Remove zero bias from QLinearConv operations."""

    op_type: ClassVar = "QLinearConv"

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


class RemoveZeroBiasFromGemm(_RemoveZeroBiasBase):
    """Remove zero bias from Gemm operations."""

    op_type: ClassVar = "Gemm"

    def pattern(self, op: ir.tape.Tape, x: ir.Value, w: ir.Value, b: ir.Value) -> ir.Value:
        return op.Gemm(x, w, b, _outputs=["out"])


# Create rule instances
remove_zero_bias_from_conv_rule = RemoveZeroBiasFromConv().rule()
remove_zero_bias_from_conv_transpose_rule = RemoveZeroBiasFromConvTranspose().rule()
remove_zero_bias_from_qlinear_conv_rule = RemoveZeroBiasFromQLinearConv().rule()
remove_zero_bias_from_gemm_rule = RemoveZeroBiasFromGemm().rule()

rules = RewriteRuleSet(
    [
        remove_zero_bias_from_conv_rule,
        remove_zero_bias_from_conv_transpose_rule,
        remove_zero_bias_from_qlinear_conv_rule,
        remove_zero_bias_from_gemm_rule,
    ]
)
