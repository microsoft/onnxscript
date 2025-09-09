# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Remove optional bias when it is all zero from Conv and related operations."""

from __future__ import annotations

import numpy as np

from onnxscript import ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class _RemoveZeroBiasBase(RewriteRuleClassBase):
    """Base class for removing zero bias from operations."""

    def __init__(self, op_type: str):
        super().__init__(remove_nodes=False)
        self.op_type = op_type

    def rewrite(self, op: ir.tape.Tape, x: ir.Value, w: ir.Value, b: ir.Value) -> ir.Value:
        """Remove the bias input from the operation."""
        return op.op(
            self.op_type,
            inputs=[x, w],  # Remove bias input
        )

    def check(self, context, x: ir.Value, w: ir.Value, b: ir.Value, **_) -> MatchResult:
        """Check if the bias is present and is all zeros."""
        del context  # Unused
        check_result = MatchResult()

        # Check if bias is a constant/initializer
        if b.const_value is None:
            return check_result.fail("Bias is not a constant/initializer.")

        # Check if bias is all zeros
        bias_array = b.const_value.numpy()
        if not np.allclose(bias_array, 0.0, atol=1e-8):
            return check_result.fail("Bias is not all zeros.")

        return check_result


class RemoveZeroBiasFromConv(_RemoveZeroBiasBase):
    """Remove zero bias from Conv operations."""

    def __init__(self):
        super().__init__("Conv")

    def pattern(self, op: ir.tape.Tape, x: ir.Value, w: ir.Value, b: ir.Value) -> ir.Value:
        return op.Conv(x, w, b, _outputs=["conv_out"])

    def check(self, context, x: ir.Value, w: ir.Value, b: ir.Value, conv_out: ir.Value, **_) -> MatchResult:
        """Check if the bias is present and is all zeros."""
        del context  # Unused
        check_result = MatchResult()

        # Check if bias is a constant/initializer
        if b.const_value is None:
            return check_result.fail("Bias is not a constant/initializer.")

        # Check if bias is all zeros
        bias_array = b.const_value.numpy()
        if not np.allclose(bias_array, 0.0, atol=1e-8):
            return check_result.fail("Bias is not all zeros.")

        return check_result

    def rewrite(self, op: ir.tape.Tape, x: ir.Value, w: ir.Value, b: ir.Value, conv_out: ir.Value) -> ir.Value:
        """Remove the bias input from the operation."""
        # Get the Conv node that produced conv_out to access its attributes
        conv_node = conv_out.producer()

        # Create new Conv with preserved attributes but without bias
        return op.op(
            "Conv",
            inputs=[x, w],  # Remove bias input
            attributes=conv_node.attributes,
            domain=conv_node.domain,
        )


class RemoveZeroBiasFromConvTranspose(_RemoveZeroBiasBase):
    """Remove zero bias from ConvTranspose operations."""

    def __init__(self):
        super().__init__("ConvTranspose")

    def pattern(self, op: ir.tape.Tape, x: ir.Value, w: ir.Value, b: ir.Value) -> ir.Value:
        return op.ConvTranspose(x, w, b, _allow_other_inputs=False, _outputs=["conv_out"])

    def rewrite(self, op: ir.tape.Tape, x: ir.Value, w: ir.Value, b: ir.Value, conv_out: ir.Value) -> ir.Value:
        """Remove the bias input from the operation."""
        # Get the ConvTranspose node that produced conv_out to access its attributes
        conv_node = conv_out.producer()

        # Create new ConvTranspose with preserved attributes but without bias
        return op.op(
            "ConvTranspose",
            inputs=[x, w],  # Remove bias input
            attributes=conv_node.attributes,
            domain=conv_node.domain,
        )


class RemoveZeroBiasFromQLinearConv(_RemoveZeroBiasBase):
    """Remove zero bias from QLinearConv operations."""

    def __init__(self):
        super().__init__("QLinearConv")

    def pattern(self, op: ir.tape.Tape, x, x_scale, x_zero_point, w, w_scale, w_zero_point,
                y_scale, y_zero_point, b: ir.Value) -> ir.Value:
        return op.QLinearConv(
            x, x_scale, x_zero_point, w, w_scale, w_zero_point,
            y_scale, y_zero_point, b, _allow_other_inputs=False, _outputs=["conv_out"]
        )

    def check(self, context, x, x_scale, x_zero_point, w, w_scale, w_zero_point,
              y_scale, y_zero_point, b: ir.Value, conv_out: ir.Value, **_) -> MatchResult:
        """Check if the bias (b) is present and is all zeros."""
        del context  # Unused
        check_result = MatchResult()

        # Check if bias is a constant/initializer
        if b.const_value is None:
            return check_result.fail("Bias is not a constant/initializer.")

        # Check if bias is all zeros
        bias_array = b.const_value.numpy()
        if not np.allclose(bias_array, 0.0, atol=1e-8):
            return check_result.fail("Bias is not all zeros.")

        return check_result

    def rewrite(self, op: ir.tape.Tape, x, x_scale, x_zero_point, w, w_scale, w_zero_point,
                y_scale, y_zero_point, b: ir.Value, conv_out: ir.Value) -> ir.Value:
        """Remove the bias input from the operation."""
        # Get the QLinearConv node that produced conv_out to access its attributes
        conv_node = conv_out.producer()

        # Create new QLinearConv with preserved attributes but without bias
        return op.op(
            "QLinearConv",
            inputs=[x, x_scale, x_zero_point, w, w_scale, w_zero_point,
                    y_scale, y_zero_point],  # Remove bias input
            attributes=conv_node.attributes,
            domain=conv_node.domain,
        )


class RemoveZeroBiasFromGemm(_RemoveZeroBiasBase):
    """Remove zero bias from Gemm operations."""

    def __init__(self):
        super().__init__("Gemm")

    def pattern(self, op: ir.tape.Tape, a: ir.Value, b: ir.Value, c: ir.Value) -> ir.Value:
        return op.Gemm(a, b, c, _allow_other_inputs=False, _outputs=["gemm_out"])

    def check(self, context, a: ir.Value, b: ir.Value, c: ir.Value, gemm_out: ir.Value, **_) -> MatchResult:
        """Check if the bias (c) is present and is all zeros."""
        del context  # Unused
        check_result = MatchResult()

        # Check if bias is a constant/initializer
        if c.const_value is None:
            return check_result.fail("Bias is not a constant/initializer.")

        # Check if bias is all zeros
        bias_array = c.const_value.numpy()
        if not np.allclose(bias_array, 0.0, atol=1e-8):
            return check_result.fail("Bias is not all zeros.")

        return check_result

    def rewrite(self, op: ir.tape.Tape, a: ir.Value, b: ir.Value, c: ir.Value, gemm_out: ir.Value) -> ir.Value:
        """Remove the bias input from the operation."""
        # Get the Gemm node that produced gemm_out to access its attributes
        gemm_node = gemm_out.producer()

        # Create new Gemm with preserved attributes but without bias
        return op.op(
            "Gemm",
            inputs=[a, b],  # Remove bias input
            attributes=gemm_node.attributes,
            domain=gemm_node.domain,
        )


# Create rule instances
remove_zero_bias_from_conv_rule = RemoveZeroBiasFromConv().rule()
remove_zero_bias_from_conv_transpose_rule = RemoveZeroBiasFromConvTranspose().rule()
remove_zero_bias_from_qlinear_conv_rule = RemoveZeroBiasFromQLinearConv().rule()
remove_zero_bias_from_gemm_rule = RemoveZeroBiasFromGemm().rule()

rules = RewriteRuleSet([
    remove_zero_bias_from_conv_rule,
    remove_zero_bias_from_conv_transpose_rule,
    remove_zero_bias_from_qlinear_conv_rule,
    remove_zero_bias_from_gemm_rule,
])
