# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the remove-Expand-before-binary-op fusion rule."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.reference
import onnx.shape_inference
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

    def test_expand_target_shape_not_constant_removed_via_output_shape(self):
        """Expand with a dynamic shape is removed when the binary op output shape
        confirms the expansion is redundant.

        x=[3, 4] has no dimension equal to 1, so Expand can only output [3, 4].
        With y=[3, 4] the binary op output shape is also [3, 4], and
        broadcast([3, 4], [3, 4]) = [3, 4] matches, so the expand is provably a
        no-op and is safely removed.
        """
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
        self.assertEqual(count, 1)
        """Expand with a symbolic x dim can be removed when y statically covers the expansion.

        x=[N], expand_shape=[3, 4], y=[3, 4]: since y provides all expand dimensions
        as known integers, the expand is redundant regardless of N's runtime value.
        """
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
        self.assertEqual(count, 1)

    def test_expand_with_symbolic_y_dim_not_removed(self):
        """Expand cannot be removed when y has a symbolic dim in a position where the
        expand is doing work and that symbolic dim cannot be verified to equal expand_d.
        """
        # x=[3], expand_shape=[4, 3], y=[M, 3].
        # At dim 0 (expand adds dim 4): x_d=1 (virtual), y_d=M (symbolic) -> can't verify.
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[3] x, float[M, 3] y) => (float[4, 3] output)
            <int64[2] shape = {4, 3}>
            {
                expanded = Expand(x, shape)
                output = Add(expanded, y)
            }
        """
        model = ir.from_onnx_text(model_text)
        count = mod.expand_before_binary_op_rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_full_optimization(self):
        oh = onnx.helper
        model_proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["x"], ["n"], start=0, end=1),
                    oh.make_node("Shape", ["x"], ["b"], start=1, end=2),
                    oh.make_node("Concat", ["n", "b"], ["shape"], axis=0),
                    oh.make_node("Expand", ["x", "shape"], ["expanded"]),
                    oh.make_node("Add", ["expanded", "y1"], ["z1"]),
                    oh.make_node("Add", ["expanded", "y2"], ["z2"]),
                    oh.make_node("Add", ["expanded", "y3"], ["z3"]),
                    oh.make_node("Add", ["z1", "z2"], ["z12"]),
                    oh.make_node("Add", ["z12", "z3"], ["z"]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["N", 1]),
                    oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, [1, "B"]),
                    oh.make_tensor_value_info("y2", onnx.TensorProto.FLOAT, [1, "B"]),
                    oh.make_tensor_value_info("y3", onnx.TensorProto.FLOAT, [1, "B"]),
                ],
                [
                    oh.make_tensor_value_info("z", onnx.TensorProto.FLOAT, ["N", "B"]),
                ],
            ),
            ir_version=11,
            opset_imports=[oh.make_opsetid("", 20)],
        )
        onnx.checker.check_model(model_proto)
        # Shape inference is required so that the Expand output carries its
        # shape annotation ([N, 1]).  Without it the rule cannot verify that
        # the expansion is redundant.
        inferred_proto = onnx.shape_inference.infer_shapes(model_proto, data_prop=True)
        model = ir.serde.deserialize_model(inferred_proto)
        count = mod.expand_before_binary_op_rules.apply_to_model(model)
        self.assertEqual(count, 3)
        self.assertEqual(len(model.graph), 5)

    def test_full_optimization_more_complex(self):
        oh = onnx.helper
        onh = onnx.numpy_helper

        model_proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["x"], ["n"], start=0, end=1),
                    oh.make_node("Shape", ["x"], ["b"], start=1, end=2),
                    oh.make_node("Concat", ["n", "b"], ["shape"], axis=0),
                    oh.make_node("Add", ["shape", "one"], ["shape1"]),
                    oh.make_node("Sub", ["shape1", "one"], ["shape2"]),
                    oh.make_node("Expand", ["x", "shape2"], ["expanded"]),
                    oh.make_node("Add", ["expanded", "y1"], ["z1"]),
                    oh.make_node("Add", ["expanded", "y2"], ["z2"]),
                    oh.make_node("Add", ["expanded", "y3"], ["z3"]),
                    oh.make_node("Add", ["z1", "z2"], ["z12"]),
                    oh.make_node("Add", ["z12", "z3"], ["z"]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["N", 1]),
                    oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, [1, "B"]),
                    oh.make_tensor_value_info("y2", onnx.TensorProto.FLOAT, [1, "B"]),
                    oh.make_tensor_value_info("y3", onnx.TensorProto.FLOAT, [1, "B"]),
                ],
                [
                    oh.make_tensor_value_info("z", onnx.TensorProto.FLOAT, ["N", "B"]),
                ],
                [onh.from_array(np.array([1], dtype=np.int64), "one")],
                # Explicit shape annotations on intermediate values (as produced by
                # shape inference or by the model creator).  These allow the rule to
                # verify that the Expand is redundant without tracing the exact
                # computation that produced the shape tensor.
                value_info=[
                    oh.make_tensor_value_info("expanded", onnx.TensorProto.FLOAT, ["N", 1]),
                    oh.make_tensor_value_info("z1", onnx.TensorProto.FLOAT, ["N", "B"]),
                    oh.make_tensor_value_info("z2", onnx.TensorProto.FLOAT, ["N", "B"]),
                    oh.make_tensor_value_info("z3", onnx.TensorProto.FLOAT, ["N", "B"]),
                ],
            ),
            ir_version=11,
            opset_imports=[oh.make_opsetid("", 20)],
        )
        onnx.checker.check_model(model_proto)
        model = ir.serde.deserialize_model(model_proto)
        count = mod.expand_before_binary_op_rules.apply_to_model(model)
        self.assertEqual(count, 3)
        self.assertEqual(len(model.graph), 5)


if __name__ == "__main__":
    unittest.main()
