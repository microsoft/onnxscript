# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for removing zero bias from Conv and related operations."""

import onnx
import onnx.parser
import onnx_ir as ir

from onnxscript.rewriter.rules.common._remove_zero_bias import (
    remove_zero_bias_from_conv_rule,
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


if __name__ == "__main__":
    test_remove_zero_bias_from_conv()
    test_conv_with_non_zero_bias_unchanged()
    print("All tests passed!")
