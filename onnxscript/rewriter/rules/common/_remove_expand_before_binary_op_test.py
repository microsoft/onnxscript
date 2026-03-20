# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the remove-Expand-before-binary-op fusion rule."""

from __future__ import annotations

import unittest

import numpy as np
import onnx.reference
import parameterized

import onnxscript.ir as ir
from onnxscript.rewriter.rules.common import _remove_expand_before_binary_op as mod


def _run_model(model: ir.Model, feeds: dict) -> list:
    """Run a model using the ONNX reference evaluator."""
    proto = ir.to_proto(model)
    ref = onnx.reference.ReferenceEvaluator(proto)
    return ref.run(None, feeds)


class RemoveExpandBeforeBinaryOpTest(unittest.TestCase):
    """Tests for _remove_expand_before_binary_op rules."""

    def _apply_and_check(
        self,
        model_text: str,
        expected_count: int,
        expected_op_types: list[str],
    ) -> ir.Model:
        """Helper: apply the rules and verify the result."""
        model = ir.from_onnx_text(model_text)
        count = mod.expand_before_binary_op_rules.apply_to_model(model)
        self.assertEqual(count, expected_count)
        actual_op_types = [node.op_type for node in model.graph]
        self.assertEqual(actual_op_types, expected_op_types)
        return model

    # ------------------------------------------------------------------
    # Cases where the Expand should be removed
    # ------------------------------------------------------------------

    @parameterized.parameterized.expand(
        [
            ("Add",),
            ("Sub",),
            ("Mul",),
            ("Div",),
        ]
    )
    def test_expand_first_input_same_shape_is_removed(self, op_type: str):
        """Expand producing same shape as input should be removed from BinaryOp."""
        model_text = f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[3, 4] x, float[3, 4] y) => (float[3, 4] output)
            <int64[2] shape = {{3, 4}}>
            {{
                expanded = Expand(x, shape)
                output = {op_type}(expanded, y)
            }}
        """
        model = self._apply_and_check(model_text, 1, [op_type])

        # Verify numerical correctness
        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(3, 4).astype(np.float32)
        original = ir.from_onnx_text(model_text)
        expected = _run_model(original, {"x": x, "y": y})
        got = _run_model(model, {"x": x, "y": y})
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-5)

    def test_expand_first_input_broadcast_covered_by_other_input(self):
        """Expand from [3, 4] to [4, 3, 4] can be removed when y has shape [4, 3, 4]."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[3, 4] x, float[4, 3, 4] y) => (float[4, 3, 4] output)
            <int64[3] shape = {4, 3, 4}>
            {
                expanded = Expand(x, shape)
                output = Add(expanded, y)
            }
        """
        model = self._apply_and_check(model_text, 1, ["Add"])

        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(4, 3, 4).astype(np.float32)
        original = ir.from_onnx_text(model_text)
        expected = _run_model(original, {"x": x, "y": y})
        got = _run_model(model, {"x": x, "y": y})
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-5)

    def test_expand_second_input_is_removed(self):
        """Expand on the second input of a binary op should be removed."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[4, 3, 4] x, float[3, 4] y) => (float[4, 3, 4] output)
            <int64[3] shape = {4, 3, 4}>
            {
                expanded = Expand(y, shape)
                output = Mul(x, expanded)
            }
        """
        model = self._apply_and_check(model_text, 1, ["Mul"])

        x = np.random.randn(4, 3, 4).astype(np.float32)
        y = np.random.randn(3, 4).astype(np.float32)
        original = ir.from_onnx_text(model_text)
        expected = _run_model(original, {"x": x, "y": y})
        got = _run_model(model, {"x": x, "y": y})
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-5)

    def test_expand_with_broadcast_compatible_other_input(self):
        """Expand from [3] to [4, 3] can be removed when y has shape [4, 1]."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[3] x, float[4, 1] y) => (float[4, 3] output)
            <int64[2] shape = {4, 3}>
            {
                expanded = Expand(x, shape)
                output = Add(expanded, y)
            }
        """
        model = self._apply_and_check(model_text, 1, ["Add"])

        x = np.random.randn(3).astype(np.float32)
        y = np.random.randn(4, 1).astype(np.float32)
        original = ir.from_onnx_text(model_text)
        expected = _run_model(original, {"x": x, "y": y})
        got = _run_model(model, {"x": x, "y": y})
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-5)

    def test_expand_sub_first_input_is_removed(self):
        """Expand on the first input of Sub should be removed."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[3, 4] x, float[3, 4] y) => (float[3, 4] output)
            <int64[2] shape = {3, 4}>
            {
                expanded = Expand(x, shape)
                output = Sub(expanded, y)
            }
        """
        model = self._apply_and_check(model_text, 1, ["Sub"])

        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(3, 4).astype(np.float32)
        original = ir.from_onnx_text(model_text)
        expected = _run_model(original, {"x": x, "y": y})
        got = _run_model(model, {"x": x, "y": y})
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-5)

    def test_expand_div_second_input_is_removed(self):
        """Expand on the second input of Div should be removed."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[4, 3, 4] x, float[3, 4] y) => (float[4, 3, 4] output)
            <int64[3] shape = {4, 3, 4}>
            {
                expanded = Expand(y, shape)
                output = Div(x, expanded)
            }
        """
        model = self._apply_and_check(model_text, 1, ["Div"])

        x = np.random.randn(4, 3, 4).astype(np.float32)
        y = (np.random.randn(3, 4).astype(np.float32) + 2.0)  # avoid division by zero
        original = ir.from_onnx_text(model_text)
        expected = _run_model(original, {"x": x, "y": y})
        got = _run_model(model, {"x": x, "y": y})
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-5)

    # ------------------------------------------------------------------
    # Cases where the Expand should NOT be removed
    # ------------------------------------------------------------------

    def test_expand_changes_output_shape_not_removed(self):
        """Expand that changes the output shape compared to direct broadcast must be kept."""
        # x has shape [3], expand to [4, 3], other is a scalar.
        # With expand: broadcast([4, 3], []) = [4, 3]
        # Without expand: broadcast([3], []) = [3]  <- different!
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[3] x) => (float[4, 3] output)
            <int64[2] shape = {4, 3}, float[1] one = {1.0}>
            {
                expanded = Expand(x, shape)
                output = Add(expanded, one)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = mod.expand_before_binary_op_rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_expand_target_shape_not_constant_not_removed(self):
        """Expand with a dynamic (non-constant) shape cannot be removed."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[3, 4] x, float[3, 4] y, int64[2] shape) => (float[3, 4] output)
            {
                expanded = Expand(x, shape)
                output = Add(expanded, y)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = mod.expand_before_binary_op_rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_expand_unknown_input_shape_not_removed(self):
        """Expand cannot be removed when the input shape is not statically known."""
        # No shape annotation on 'x'
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, float[3, 4] y) => (float[3, 4] output)
            <int64[2] shape = {3, 4}>
            {
                expanded = Expand(x, shape)
                output = Add(expanded, y)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = mod.expand_before_binary_op_rules.apply_to_model(model)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
