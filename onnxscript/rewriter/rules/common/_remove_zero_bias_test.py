# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for removing zero bias from Conv and related operations."""

import unittest
from typing import Optional

import onnx_ir as ir

from onnxscript.rewriter import testing
from onnxscript.rewriter.rules.common._remove_zero_bias import (
    remove_zero_bias_from_conv_rule,
    remove_zero_bias_from_conv_transpose_rule,
    remove_zero_bias_from_gemm_rule,
    remove_zero_bias_from_qlinear_conv_rule,
)


def _apply_rule_and_check_optimization(
    model: ir.Model,
    rule,
    expected_count: int,
    target_op_type: str,
    expected_inputs_after: int,
    expected_attributes: Optional[dict] = None,
) -> None:
    """Helper function to test bias removal rules."""
    # Make a copy for comparison
    original_model = ir.from_proto(ir.to_proto(model))

    # Get original attributes for comparison
    original_target_node = None
    for node in original_model.graph:
        if node.op_type == target_op_type:
            original_target_node = node
            break

    # Apply the rule
    count = rule.apply_to_model(model)

    # Check that the rule was applied the expected number of times
    assert count == expected_count, f"Expected {expected_count} applications, got {count}"

    # Check that the target node has the expected number of inputs
    target_node = None
    for node in model.graph:
        if node.op_type == target_op_type:
            target_node = node
            break

    assert target_node is not None, f"{target_op_type} node not found"
    assert len(target_node.inputs) == expected_inputs_after, (
        f"Expected {expected_inputs_after} inputs after optimization, "
        f"got {len(target_node.inputs)}"
    )

    # Check that attributes are preserved if the rule was applied
    if expected_count > 0 and original_target_node is not None:
        # All original attributes should be preserved
        for attr_name, attr_value in original_target_node.attributes.items():
            assert attr_name in target_node.attributes, f"Attribute {attr_name} was lost"
            original_value = attr_value.value
            new_value = target_node.attributes[attr_name].value
            assert new_value == original_value, (
                f"Attribute {attr_name} value changed from {original_value} to {new_value}"
            )

    # Check specific expected attributes if provided
    if expected_attributes:
        for attr_name, expected_value in expected_attributes.items():
            assert attr_name in target_node.attributes, (
                f"Expected attribute {attr_name} not found"
            )
            actual_attr = target_node.attributes[attr_name]
            actual_value = actual_attr.value
            assert actual_value == expected_value, (
                f"Expected attribute {attr_name} to be {expected_value}, got {actual_value}"
            )

    # Compare outputs to ensure correctness (only for supported input types)
    if expected_count > 0:
        try:
            # Generate random inputs for the model using the existing testing utility
            original_model_proto = ir.to_proto(original_model)
            inputs = testing.generate_random_inputs(original_model_proto)
            testing.assert_numerically_equal(original_model, model, inputs)
        except ValueError as e:
            if "Not implemented for input type" in str(e):
                # Skip numerical comparison for unsupported input types
                # The structural checks above are sufficient for these cases
                pass
            else:
                raise


class RemoveZeroBiasTest(unittest.TestCase):
    """Test class for remove zero bias rules."""

    def test_remove_zero_bias_from_conv(self):
        """Test that zero bias is removed from Conv operations."""
        # Create a simple Conv with zero bias using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 2, 4, 4] x) => (float[1, 2, 2, 2] y)
            {
                weight = Constant <value = float[2, 2, 3, 3] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}>()
                bias = Constant <value = float[2] {0, 0}>()
                y = Conv(x, weight, bias)
            }
            """
        )

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_conv_rule,
            expected_count=1,
            target_op_type="Conv",
            expected_inputs_after=2,
        )

    def test_conv_with_non_zero_bias_unchanged(self):
        """Test that Conv with non-zero bias is not modified."""
        # Create a Conv with non-zero bias using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 2, 4, 4] x) => (float[1, 2, 2, 2] y)
            {
                weight = Constant <value = float[2, 2, 3, 3] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}>()
                bias = Constant <value = float[2] {1, 1}>()
                y = Conv(x, weight, bias)
            }
            """
        )

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_conv_rule,
            expected_count=0,
            target_op_type="Conv",
            expected_inputs_after=3,
        )

    def test_remove_zero_bias_from_conv_transpose(self):
        """Test that zero bias is removed from ConvTranspose operations."""
        # Create a ConvTranspose with zero bias using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 2, 2, 2] x) => (float[1, 2, 4, 4] y)
            {
                weight = Constant <value = float[2, 2, 3, 3] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}>()
                bias = Constant <value = float[2] {0, 0}>()
                y = ConvTranspose(x, weight, bias)
            }
            """
        )

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_conv_transpose_rule,
            expected_count=1,
            target_op_type="ConvTranspose",
            expected_inputs_after=2,
        )

    def test_conv_transpose_with_non_zero_bias_unchanged(self):
        """Test that ConvTranspose with non-zero bias is not modified."""
        # Create a ConvTranspose with non-zero bias using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 2, 2, 2] x) => (float[1, 2, 4, 4] y)
            {
                weight = Constant <value = float[2, 2, 3, 3] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}>()
                bias = Constant <value = float[2] {1, 1}>()
                y = ConvTranspose(x, weight, bias)
            }
            """
        )

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_conv_transpose_rule,
            expected_count=0,
            target_op_type="ConvTranspose",
            expected_inputs_after=3,
        )

    def test_remove_zero_bias_from_gemm(self):
        """Test that zero bias is removed from Gemm operations."""
        # Create a Gemm with zero bias using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 3] a) => (float[2, 4] y)
            {
                b = Constant <value = float[3, 4] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}>()
                c = Constant <value = float[4] {0, 0, 0, 0}>()
                y = Gemm(a, b, c)
            }
            """
        )

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_gemm_rule,
            expected_count=1,
            target_op_type="Gemm",
            expected_inputs_after=2,
        )

    def test_gemm_with_non_zero_bias_unchanged(self):
        """Test that Gemm with non-zero bias is not modified."""
        # Create a Gemm with non-zero bias using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 3] a) => (float[2, 4] y)
            {
                b = Constant <value = float[3, 4] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}>()
                c = Constant <value = float[4] {1, 0, 0, 1}>()
                y = Gemm(a, b, c)
            }
            """
        )

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_gemm_rule,
            expected_count=0,
            target_op_type="Gemm",
            expected_inputs_after=3,
        )

    def test_remove_zero_bias_from_qlinear_conv(self):
        """Test that zero bias is removed from QLinearConv operations."""
        # Create a QLinearConv with zero bias using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (uint8[1, 2, 4, 4] x) => (uint8[1, 2, 2, 2] y)
            {
                x_scale = Constant <value = float {0.1}>()
                x_zero_point = Constant <value = uint8 {128}>()
                weight = Constant <value = uint8[2, 2, 3, 3] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}>()
                w_scale = Constant <value = float {0.05}>()
                w_zero_point = Constant <value = uint8 {64}>()
                y_scale = Constant <value = float {0.2}>()
                y_zero_point = Constant <value = uint8 {192}>()
                bias = Constant <value = int32[2] {0, 0}>()
                y = QLinearConv(x, x_scale, x_zero_point, weight, w_scale, w_zero_point, y_scale, y_zero_point, bias)
            }
            """
        )

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_qlinear_conv_rule,
            expected_count=1,
            target_op_type="QLinearConv",
            expected_inputs_after=8,
        )

    def test_remove_zero_bias_from_conv_with_attributes(self):
        """Test that zero bias is removed from Conv operations and attributes are preserved."""
        # Create a Conv with zero bias and various attributes using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 2, 6, 6] x) => (float[1, 2, 2, 2] y)
            {
                weight = Constant <value = float[2, 2, 3, 3] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}>()
                bias = Constant <value = float[2] {0, 0}>()
                y = Conv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [2, 2]> (x, weight, bias)
            }
            """
        )

        expected_attributes = {
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [3, 3],
            "pads": [0, 0, 0, 0],
            "strides": [2, 2],
        }

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_conv_rule,
            expected_count=1,
            target_op_type="Conv",
            expected_inputs_after=2,
            expected_attributes=expected_attributes,
        )

    def test_remove_zero_bias_from_conv_transpose_with_attributes(self):
        """Test that zero bias is removed from ConvTranspose operations and attributes are preserved."""
        # Create a ConvTranspose with zero bias and various attributes using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[1, 2, 2, 2] x) => (float[1, 2, 6, 6] y)
            {
                weight = Constant <value = float[2, 2, 3, 3] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}>()
                bias = Constant <value = float[2] {0, 0}>()
                y = ConvTranspose <dilations = [1, 1], group = 1, kernel_shape = [3, 3], output_padding = [0, 0], pads = [0, 0, 0, 0], strides = [2, 2]> (x, weight, bias)
            }
            """
        )

        expected_attributes = {
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [3, 3],
            "output_padding": [0, 0],
            "pads": [0, 0, 0, 0],
            "strides": [2, 2],
        }

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_conv_transpose_rule,
            expected_count=1,
            target_op_type="ConvTranspose",
            expected_inputs_after=2,
            expected_attributes=expected_attributes,
        )

    def test_remove_zero_bias_from_gemm_with_attributes(self):
        """Test that zero bias is removed from Gemm operations and attributes are preserved."""
        # Create a Gemm with zero bias and various attributes using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[2, 3] a) => (float[2, 4] y)
            {
                b = Constant <value = float[4, 3] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}>()
                c = Constant <value = float[4] {0, 0, 0, 0}>()
                y = Gemm <alpha = 2.0, beta = 1.0, transA = 0, transB = 1> (a, b, c)
            }
            """
        )

        expected_attributes = {
            "alpha": 2.0,
            "beta": 1.0,
            "transA": 0,
            "transB": 1,
        }

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_gemm_rule,
            expected_count=1,
            target_op_type="Gemm",
            expected_inputs_after=2,
            expected_attributes=expected_attributes,
        )

    def test_remove_zero_bias_from_qlinear_conv_with_attributes(self):
        """Test that zero bias is removed from QLinearConv operations and attributes are preserved."""
        # Create a QLinearConv with zero bias and various attributes using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (uint8[1, 2, 6, 6] x) => (uint8[1, 2, 2, 2] y)
            {
                x_scale = Constant <value = float {0.1}>()
                x_zero_point = Constant <value = uint8 {128}>()
                weight = Constant <value = uint8[2, 2, 3, 3] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}>()
                w_scale = Constant <value = float {0.05}>()
                w_zero_point = Constant <value = uint8 {64}>()
                y_scale = Constant <value = float {0.2}>()
                y_zero_point = Constant <value = uint8 {192}>()
                bias = Constant <value = int32[2] {0, 0}>()
                y = QLinearConv <dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [2, 2]> (x, x_scale, x_zero_point, weight, w_scale, w_zero_point, y_scale, y_zero_point, bias)
            }
            """
        )

        expected_attributes = {
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [3, 3],
            "pads": [0, 0, 0, 0],
            "strides": [2, 2],
        }

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_qlinear_conv_rule,
            expected_count=1,
            target_op_type="QLinearConv",
            expected_inputs_after=8,
            expected_attributes=expected_attributes,
        )

    def test_qlinear_conv_with_non_zero_bias_unchanged(self):
        """Test that QLinearConv with non-zero bias is not modified."""
        # Create a QLinearConv with non-zero bias using ONNX text format
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (uint8[1, 2, 4, 4] x) => (uint8[1, 2, 2, 2] y)
            {
                x_scale = Constant <value = float {0.1}>()
                x_zero_point = Constant <value = uint8 {128}>()
                weight = Constant <value = uint8[2, 2, 3, 3] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}>()
                w_scale = Constant <value = float {0.05}>()
                w_zero_point = Constant <value = uint8 {64}>()
                y_scale = Constant <value = float {0.2}>()
                y_zero_point = Constant <value = uint8 {192}>()
                bias = Constant <value = int32[2] {1, 1}>()
                y = QLinearConv(x, x_scale, x_zero_point, weight, w_scale, w_zero_point, y_scale, y_zero_point, bias)
            }
            """
        )

        _apply_rule_and_check_optimization(
            model,
            remove_zero_bias_from_qlinear_conv_rule,
            expected_count=0,
            target_op_type="QLinearConv",
            expected_inputs_after=9,
        )


if __name__ == "__main__":
    unittest.main()
