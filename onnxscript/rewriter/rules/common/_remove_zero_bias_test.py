# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for removing zero bias from Conv and related operations."""

import onnx
import onnx.parser
import onnx_ir as ir

from onnxscript.rewriter.rules.common._remove_zero_bias import (
    remove_zero_bias_from_conv_rule,
    remove_zero_bias_from_conv_transpose_rule,
    remove_zero_bias_from_gemm_rule,
    remove_zero_bias_from_qlinear_conv_rule,
)


def test_remove_zero_bias_from_conv():
    """Test that zero bias is removed from Conv operations."""
    # Create a simple Conv with zero bias using ONNX parser
    model_proto = onnx.parser.parse_model(
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

    # Convert to IR model
    model = ir.serde.deserialize_model(model_proto)

    # Apply the rule
    count = remove_zero_bias_from_conv_rule.apply_to_model(model)

    # Check that the rule was applied
    assert count == 1, f"Expected 1 application, got {count}"

    # Check that bias input was removed
    conv_node = None
    for node in model.graph:
        if node.op_type == "Conv":
            conv_node = node
            break

    assert conv_node is not None, "Conv node not found"
    assert len(conv_node.inputs) == 2, f"Expected 2 inputs after optimization, got {len(conv_node.inputs)}"


def test_conv_with_non_zero_bias_unchanged():
    """Test that Conv with non-zero bias is not modified."""
    # Create a Conv with non-zero bias using ONNX parser
    model_proto = onnx.parser.parse_model(
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

    # Convert to IR model
    model = ir.serde.deserialize_model(model_proto)

    # Apply the rule
    count = remove_zero_bias_from_conv_rule.apply_to_model(model)

    # Check that the rule was NOT applied
    assert count == 0, f"Expected 0 applications, got {count}"

    # Check that bias input is still present
    conv_node = None
    for node in model.graph:
        if node.op_type == "Conv":
            conv_node = node
            break

    assert conv_node is not None, "Conv node not found"
    assert len(conv_node.inputs) == 3, f"Expected 3 inputs, got {len(conv_node.inputs)}"


def test_remove_zero_bias_from_conv_transpose():
    """Test that zero bias is removed from ConvTranspose operations."""
    # Create a ConvTranspose with zero bias using ONNX parser
    model_proto = onnx.parser.parse_model(
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

    # Convert to IR model
    model = ir.serde.deserialize_model(model_proto)

    # Apply the rule
    count = remove_zero_bias_from_conv_transpose_rule.apply_to_model(model)

    # Check that the rule was applied
    assert count == 1, f"Expected 1 application, got {count}"

    # Check that bias input was removed
    conv_node = None
    for node in model.graph:
        if node.op_type == "ConvTranspose":
            conv_node = node
            break

    assert conv_node is not None, "ConvTranspose node not found"
    assert len(conv_node.inputs) == 2, f"Expected 2 inputs after optimization, got {len(conv_node.inputs)}"


def test_conv_transpose_with_non_zero_bias_unchanged():
    """Test that ConvTranspose with non-zero bias is not modified."""
    # Create a ConvTranspose with non-zero bias using ONNX parser
    model_proto = onnx.parser.parse_model(
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

    # Convert to IR model
    model = ir.serde.deserialize_model(model_proto)

    # Apply the rule
    count = remove_zero_bias_from_conv_transpose_rule.apply_to_model(model)

    # Check that the rule was NOT applied
    assert count == 0, f"Expected 0 applications, got {count}"

    # Check that bias input is still present
    conv_node = None
    for node in model.graph:
        if node.op_type == "ConvTranspose":
            conv_node = node
            break

    assert conv_node is not None, "ConvTranspose node not found"
    assert len(conv_node.inputs) == 3, f"Expected 3 inputs, got {len(conv_node.inputs)}"


def test_remove_zero_bias_from_gemm():
    """Test that zero bias is removed from Gemm operations."""
    # Create a Gemm with zero bias using ONNX parser
    model_proto = onnx.parser.parse_model(
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

    # Convert to IR model
    model = ir.serde.deserialize_model(model_proto)

    # Apply the rule
    count = remove_zero_bias_from_gemm_rule.apply_to_model(model)

    # Check that the rule was applied
    assert count == 1, f"Expected 1 application, got {count}"

    # Check that bias input was removed
    gemm_node = None
    for node in model.graph:
        if node.op_type == "Gemm":
            gemm_node = node
            break

    assert gemm_node is not None, "Gemm node not found"
    assert len(gemm_node.inputs) == 2, f"Expected 2 inputs after optimization, got {len(gemm_node.inputs)}"


def test_gemm_with_non_zero_bias_unchanged():
    """Test that Gemm with non-zero bias is not modified."""
    # Create a Gemm with non-zero bias using ONNX parser
    model_proto = onnx.parser.parse_model(
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

    # Convert to IR model
    model = ir.serde.deserialize_model(model_proto)

    # Apply the rule
    count = remove_zero_bias_from_gemm_rule.apply_to_model(model)

    # Check that the rule was NOT applied
    assert count == 0, f"Expected 0 applications, got {count}"

    # Check that bias input is still present
    gemm_node = None
    for node in model.graph:
        if node.op_type == "Gemm":
            gemm_node = node
            break

    assert gemm_node is not None, "Gemm node not found"
    assert len(gemm_node.inputs) == 3, f"Expected 3 inputs, got {len(gemm_node.inputs)}"


def test_remove_zero_bias_from_qlinear_conv():
    """Test that zero bias is removed from QLinearConv operations."""
    # Create a QLinearConv with zero bias using ONNX parser
    model_proto = onnx.parser.parse_model(
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

    # Convert to IR model
    model = ir.serde.deserialize_model(model_proto)

    # Apply the rule
    count = remove_zero_bias_from_qlinear_conv_rule.apply_to_model(model)

    # Check that the rule was applied
    assert count == 1, f"Expected 1 application, got {count}"

    # Check that bias input was removed
    qconv_node = None
    for node in model.graph:
        if node.op_type == "QLinearConv":
            qconv_node = node
            break

    assert qconv_node is not None, "QLinearConv node not found"
    assert len(qconv_node.inputs) == 8, f"Expected 8 inputs after optimization, got {len(qconv_node.inputs)}"


def test_qlinear_conv_with_non_zero_bias_unchanged():
    """Test that QLinearConv with non-zero bias is not modified."""
    # Create a QLinearConv with non-zero bias using ONNX parser
    model_proto = onnx.parser.parse_model(
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

    # Convert to IR model
    model = ir.serde.deserialize_model(model_proto)

    # Apply the rule
    count = remove_zero_bias_from_qlinear_conv_rule.apply_to_model(model)

    # Check that the rule was NOT applied
    assert count == 0, f"Expected 0 applications, got {count}"

    # Check that bias input is still present
    qconv_node = None
    for node in model.graph:
        if node.op_type == "QLinearConv":
            qconv_node = node
            break

    assert qconv_node is not None, "QLinearConv node not found"
    assert len(qconv_node.inputs) == 9, f"Expected 9 inputs, got {len(qconv_node.inputs)}"


if __name__ == "__main__":
    test_remove_zero_bias_from_conv()
    test_conv_with_non_zero_bias_unchanged()
    test_remove_zero_bias_from_conv_transpose()
    test_conv_transpose_with_non_zero_bias_unchanged()
    test_remove_zero_bias_from_gemm()
    test_gemm_with_non_zero_bias_unchanged()
    test_remove_zero_bias_from_qlinear_conv()
    test_qlinear_conv_with_non_zero_bias_unchanged()
    print("All tests passed!")
