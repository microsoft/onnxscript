# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.reference
import onnx_ir as ir

from onnxscript.rewriter.rules.common import _layout_optimizations as layout


def _evaluate(model: ir.Model, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    return onnx.reference.ReferenceEvaluator(ir.to_proto(model)).run(None, feeds)


def _apply(model_text: str) -> tuple[ir.Model, ir.Model, int]:
    original = ir.from_onnx_text(model_text)
    rewritten = ir.from_onnx_text(model_text)
    count = layout.rules.apply_to_model(rewritten)
    onnx.checker.check_model(ir.to_proto(rewritten))
    return original, rewritten, count


def _apply_inferred_rule(model_text: str, rule) -> tuple[ir.Model, ir.Model, int]:
    parsed = ir.from_onnx_text(model_text)
    inferred_proto = onnx.shape_inference.infer_shapes(ir.to_proto(parsed))
    original = ir.from_proto(inferred_proto)
    rewritten = ir.from_proto(inferred_proto)
    count = rule.apply_to_model(rewritten)
    onnx.checker.check_model(ir.to_proto(rewritten))
    return original, rewritten, count


class LayoutOptimizationsTest(unittest.TestCase):
    def assert_equivalent(self, original, rewritten, feeds):
        expected = _evaluate(original, feeds)
        actual = _evaluate(rewritten, feeds)
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=0, atol=0, equal_nan=True)
            if np.issubdtype(got.dtype, np.floating):
                np.testing.assert_array_equal(np.signbit(got), np.signbit(want))

    def test_move_shape_op_across_elementwise(self):
        cases = [
            (
                "squeeze_add",
                """
                <ir_version: 10, opset_import: ["" : 18]>
                g (float[1,3] x, float[1,3] y) => (float[3] z)
                <int64[1] axes = {0}> {
                    a = Squeeze(x, axes)
                    b = Squeeze(y, axes)
                    z = Add(a, b)
                }
                """,
                ["Add", "Squeeze"],
            ),
            (
                "unsqueeze_mul",
                """
                <ir_version: 10, opset_import: ["" : 18]>
                g (float[3] x, float[3] y) => (float[1,3] z)
                <int64[1] axes = {0}> {
                    a = Unsqueeze(x, axes)
                    b = Unsqueeze(y, axes)
                    z = Mul(a, b)
                }
                """,
                ["Mul", "Unsqueeze"],
            ),
            (
                "squeeze_sub_unsqueeze",
                """
                <ir_version: 10, opset_import: ["" : 18]>
                g (float[1] x) => (float[1] z)
                <int64[1] axes = {0}, float y = {2.0}> {
                    a = Squeeze(x, axes)
                    b = Sub(a, y)
                    z = Unsqueeze(b, axes)
                }
                """,
                ["Unsqueeze", "Sub"],
            ),
        ]
        for name, model_text, expected_ops in cases:
            with self.subTest(name=name):
                original, rewritten, count = _apply(model_text)
                self.assertEqual(count, 1)
                self.assertEqual([node.op_type for node in rewritten.graph], expected_ops)
                feeds = {
                    value.name: np.random.randn(*tuple(int(d) for d in value.shape)).astype(
                        np.float32
                    )
                    for value in original.graph.inputs
                }
                expected = _evaluate(original, feeds)
                actual = _evaluate(rewritten, feeds)
                np.testing.assert_array_equal(actual[0], expected[0])

    def test_squeeze_binary_requires_proven_unit_dimension(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[N] x) => (float[1] z)
            <int64[1] axes = {0}, float y = {2.0}> {
                a = Squeeze(x, axes)
                b = Div(a, y)
                z = Unsqueeze(b, axes)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_generated_squeeze_binary_rules(self):
        for op_type in ("Add", "Div", "Mul", "Sub"):
            model_text = f"""
                <ir_version: 10, opset_import: ["" : 18]>
                g (float[1] x, float other) => (float[1] y)
                <int64[1] axes = {{0}}> {{
                    squeezed = Squeeze(x, axes)
                    result = {op_type}(squeezed, other)
                    y = Unsqueeze(result, axes)
                }}
            """
            with self.subTest(op_type=op_type):
                original, rewritten, count = _apply(model_text)
                self.assertEqual(count, 1)
                self.assertEqual(
                    [node.op_type for node in rewritten.graph],
                    ["Unsqueeze", op_type],
                )
                self.assert_equivalent(
                    original,
                    rewritten,
                    {
                        "x": np.array([3.0], dtype=np.float32),
                        "other": np.array(2.0, dtype=np.float32),
                    },
                )

    def test_squeeze_add_rank_mismatch_rejected(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,1,2] x, float[2,1] y) => (float[2,2] z)
            <int64[1] axes = {1}> {
                a = Squeeze(x, axes)
                b = Squeeze(y, axes)
                z = Add(a, b)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_unsqueeze_mul_rank_mismatch_rejected(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3] x, float[3] y) => (float[2,3,3] z)
            <int64[1] axes = {1}> {
                a = Unsqueeze(x, axes)
                b = Unsqueeze(y, axes)
                z = Mul(a, b)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_swap_unsqueeze_transpose_multiple_negative_axes(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3] x) => (float[1,3,2,1] y)
            <int64[2] axes = {-4,-1}> {
                u = Unsqueeze(x, axes)
                y = Transpose(u) <perm = [0,2,1,3]>
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual(
            [n.op_type for n in rewritten.graph], ["Constant", "Transpose", "Unsqueeze"]
        )
        feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_transpose_scalar_gather_moves_before_transpose(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3,4] x) => (float[4,2] y)
            <int64 index = {1}> {
                t = Transpose(x) <perm = [1,2,0]>
                y = Gather(t, index) <axis = -2>
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Gather", "Transpose"])
        feeds = {"x": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_transpose_gather_supports_default_axis_and_permutation(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3,4] x) => (float[3,2] y)
            <int64 index = {1}> {
                t = Transpose(x)
                y = Gather(t, index)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Gather", "Transpose"])
        feeds = {"x": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_transpose_gather_default_attribute_variants(self):
        cases = [
            (
                "default_axis",
                "t = Transpose(x) <perm = [1,2,0]>\n                y = Gather(t, index)",
                "float[4,2]",
            ),
            (
                "default_permutation",
                "t = Transpose(x)\n                y = Gather(t, index) <axis = -2>",
                "float[4,2]",
            ),
        ]
        for name, body, output_type in cases:
            with self.subTest(name=name):
                model_text = f"""
                    <ir_version: 10, opset_import: ["" : 18]>
                    g (float[2,3,4] x) => ({output_type} y)
                    <int64 index = {{1}}> {{
                        {body}
                    }}
                """
                original, rewritten, count = _apply(model_text)
                self.assertEqual(count, 1)
                self.assertEqual(
                    [node.op_type for node in rewritten.graph], ["Gather", "Transpose"]
                )
                self.assert_equivalent(
                    original,
                    rewritten,
                    {"x": np.arange(24, dtype=np.float32).reshape(2, 3, 4)},
                )

    def test_transpose_of_unit_axes_becomes_reshape_with_zero_dim(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[0,1,3,1] x) => (float[0,3,1,1] y) {
                y = Transpose(x) <perm = [0,2,1,3]>
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Constant", "Reshape"])
        feeds = {"x": np.empty((0, 1, 3, 1), dtype=np.float32)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_transpose_equal_reshape_is_valid_in_opset13(self):
        model_text = """
            <ir_version: 8, opset_import: ["" : 13]>
            g (float[2,1,3] x) => (float[2,3,1] y) {
                y = Transpose(x) <perm = [0,2,1]>
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        reshape = list(rewritten.graph)[1]
        self.assertEqual(reshape.op_type, "Reshape")
        self.assertNotIn("allowzero", reshape.attributes)
        feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 1, 3)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_merge_multi_axis_unsqueezes(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3] x) => (float[1,2,1,3,1] y)
            <int64[2] a = {0,2}, int64[1] b = {-1}> {
                u = Unsqueeze(x, a)
                y = Unsqueeze(u, b)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_unsqueeze_squeeze_subset_preserves_shared_intermediate(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,1,3] x) => (float[2,3] y, float[2,1,1,3] side)
            <int64[1] ua = {1}, int64[2] sa = {1,2}> {
                side = Unsqueeze(x, ua)
                y = Squeeze(side, sa)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual(
            [n.op_type for n in rewritten.graph], ["Unsqueeze", "Constant", "Squeeze"]
        )
        feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 1, 3)}
        expected = _evaluate(original, feeds)
        actual = _evaluate(rewritten, feeds)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)

    def test_reshape_squeeze_fuses_and_allowzero_one_rejects(self):
        positive = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3] x) => (float[2,3] y)
            <int64[3] shape = {0,1,-1}, int64[1] axes = {-2}> {
                r = Reshape(x, shape)
                y = Squeeze(r, axes)
            }
        """
        original, rewritten, count = _apply(positive)
        self.assertEqual(count, 2)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Identity"])
        feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

        rejected = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[0,6] x) => (float[0,2,3] y)
            <int64[4] shape = {0,1,2,3}, int64[1] axes = {1}> {
                r = Reshape(x, shape) <allowzero = 1>
                y = Squeeze(r, axes)
            }
        """
        _, _, count = _apply(rejected)
        self.assertEqual(count, 0)

    def test_reshape_squeeze_empty_axes_is_not_fused(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,2] x) => (float[2,1] y)
            <int64[2] shape = {2,1}, int64[0] axes = {}> {
                r = Reshape(x, shape)
                y = Squeeze(r, axes)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_no_op_reshape_symbolic_and_zero_sized(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[N,0,3] x) => (float[N,0,3] y)
            <int64[3] shape = {0,0,3}> {
                y = Reshape(x, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Identity"])
        feeds = {"x": np.empty((2, 0, 3), dtype=np.float32)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_no_op_reshape_rejects_unknown_dimensions(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[?,?] x) => (float[?,?] y)
            <int64[2] shape = {-1,1}> {
                y = Reshape(x, shape)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_no_op_reshape_accepts_unambiguous_symbolic_inferred_dimension(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[N,3] x) => (float[N,3] y)
            <int64[2] shape = {-1,3}> {
                y = Reshape(x, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Identity"])
        for leading in (0, 2):
            feeds = {"x": np.arange(leading * 3, dtype=np.float32).reshape(leading, 3)}
            np.testing.assert_array_equal(
                _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
            )

    def test_no_op_reshape_rejects_ambiguous_symbolic_inference(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[N,3] x) => (float[N,3] y)
            <int64[2] shape = {0,-1}> {
                y = Reshape(x, shape)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_no_op_reshape_allowzero_literal_zero(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[0,3] x) => (float[0,3] y)
            <int64[2] shape = {0,3}> {
                y = Reshape(x, shape) <allowzero = 1>
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([node.op_type for node in rewritten.graph], ["Identity"])
        feeds = {"x": np.empty((0, 3), dtype=np.float32)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_shape_based_reshape_becomes_unsqueeze(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[N,3] x) => (float[1,N,3,1] y)
            <int64[4] shape = {1,-1,3,1}> {
                y = Reshape(x, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Constant", "Unsqueeze"])
        feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_shape_based_reshape_becomes_squeeze(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,2,1,3] x) => (float[2,3] y)
            <int64[2] shape = {2,3}> {
                y = Reshape(x, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Constant", "Squeeze"])
        self.assert_equivalent(
            original,
            rewritten,
            {"x": np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3)},
        )

    def test_inverse_squeeze_unsqueeze_becomes_identity(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,1,3] x) => (float[2,1,3] y)
            <int64[1] squeeze_axes = {-2}, int64[1] unsqueeze_axes = {1}> {
                squeezed = Squeeze(x, squeeze_axes)
                y = Unsqueeze(squeezed, unsqueeze_axes)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Identity"])
        self.assert_equivalent(
            original,
            rewritten,
            {"x": np.arange(6, dtype=np.float32).reshape(2, 1, 3)},
        )

    def test_remove_unsqueeze_before_reshape_without_zero_copy(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3] x) => (float[3,2] y)
            <int64[1] axes = {0}, int64[2] shape = {3,2}> {
                u = Unsqueeze(x, axes)
                y = Reshape(u, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Reshape"])
        feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_remove_shape_before_reshape_rejects_zero_copy(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3] x) => (float[1,6] y)
            <int64[1] axes = {0}, int64[2] shape = {0,-1}> {
                u = Unsqueeze(x, axes)
                y = Reshape(u, shape)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_remove_squeeze_before_reshape(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,2,3] x) => (float[3,2] y)
            <int64[1] axes = {0}, int64[2] shape = {3,2}> {
                squeezed = Squeeze(x, axes)
                y = Reshape(squeezed, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Reshape"])
        self.assert_equivalent(
            original,
            rewritten,
            {"x": np.arange(6, dtype=np.float32).reshape(1, 2, 3)},
        )

    def test_equal_input_reshapes_move_after_binary(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3] x, float[2,3] z) => (float[3,2] y)
            <int64[2] shape = {3,2}> {
                a = Reshape(x, shape)
                b = Reshape(z, shape)
                y = Sub(a, b)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Sub", "Reshape"])
        feeds = {
            "x": np.arange(6, dtype=np.float32).reshape(2, 3),
            "z": np.ones((2, 3), dtype=np.float32),
        }
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_equal_input_reshapes_reject_unknown_source_shapes(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[?,?] x, float[?,?] z) => (float[6] y)
            <int64[1] shape = {6}> {
                a = Reshape(x, shape)
                b = Reshape(z, shape)
                y = Add(a, b)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_equal_input_reshapes_preserve_allowzero(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[0,3] x, float[0,3] z) => (float[0,3] y)
            <int64[2] shape = {0,3}> {
                a = Reshape(x, shape) <allowzero = 1>
                b = Reshape(z, shape) <allowzero = 1>
                y = Add(a, b)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Add", "Reshape"])
        reshape = list(rewritten.graph)[1]
        self.assertEqual(reshape.attributes["allowzero"].as_int(), 1)
        feeds = {
            "x": np.empty((0, 3), dtype=np.float32),
            "z": np.empty((0, 3), dtype=np.float32),
        }
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_generated_binary_reshape_families(self):
        bool_ops = {"And", "Or", "Xor"}
        integer_ops = {"BitwiseAnd", "BitwiseOr", "BitwiseXor"}
        ops = (
            "Add",
            "And",
            "BitwiseAnd",
            "BitwiseOr",
            "BitwiseXor",
            "Div",
            "Max",
            "Mean",
            "Min",
            "Mul",
            "Or",
            "Sub",
            "Sum",
            "Xor",
        )
        for op_type in ops:
            if op_type in bool_ops:
                tensor_type = "bool"
                x = np.array([[True, False], [False, True]])
                y = np.array([[True, True], [False, False]])
            elif op_type in integer_ops:
                tensor_type = "int32"
                x = np.array([[1, 2], [4, 8]], dtype=np.int32)
                y = np.array([[3, 1], [6, 2]], dtype=np.int32)
            else:
                tensor_type = "float"
                x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
                y = np.array([[2.0, 1.0], [4.0, 2.0]], dtype=np.float32)

            equal_reshapes = f"""
                <ir_version: 10, opset_import: ["" : 18]>
                g ({tensor_type}[2,2] x, {tensor_type}[2,2] y)
                    => ({tensor_type}[4] z)
                <int64[1] shape = {{4}}> {{
                    left = Reshape(x, shape)
                    right = Reshape(y, shape)
                    z = {op_type}(left, right)
                }}
            """
            with self.subTest(family="equal_reshapes", op_type=op_type):
                original, rewritten, count = _apply(equal_reshapes)
                self.assertEqual(count, 1)
                self.assertEqual(
                    [node.op_type for node in rewritten.graph], [op_type, "Reshape"]
                )
                self.assert_equivalent(original, rewritten, {"x": x, "y": y})

            for side in ("left", "right"):
                if side == "left":
                    body = (
                        "reshaped = Reshape(x, shape)\n"
                        f"                        inner = {op_type}(reshaped, y)"
                    )
                else:
                    body = (
                        "reshaped = Reshape(y, shape)\n"
                        f"                        inner = {op_type}(x, reshaped)"
                    )
                around_binary = f"""
                    <ir_version: 10, opset_import: ["" : 18]>
                    g ({tensor_type}[2,2] x, {tensor_type}[2,2] y)
                        => ({tensor_type}[2,2] z)
                    <int64[2] shape = {{2,2}}> {{
                        {body}
                        z = Reshape(inner, shape)
                    }}
                """
                with self.subTest(
                    family="identity_reshapes_around_binary",
                    op_type=op_type,
                    side=side,
                ):
                    rule = layout._RemoveIdentityReshapesAroundBinary.rule(
                        op_type, side == "left"
                    )
                    original, rewritten, count = _apply_inferred_rule(around_binary, rule)
                    self.assertEqual(count, 1)
                    self.assertEqual([node.op_type for node in rewritten.graph], [op_type])
                    self.assert_equivalent(original, rewritten, {"x": x, "y": y})

    def test_reduce_reshape_becomes_keepdims_zero(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3,4] x) => (float[2,4] y)
            <int64[1] axes = {1}, int64[2] shape = {2,4}> {
                r = ReduceSum(x, axes) <keepdims = 1>
                y = Reshape(r, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["ReduceSum"])
        feeds = {"x": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_reduce_attribute_axes_reshape_becomes_keepdims_zero(self):
        model_text = """
            <ir_version: 9, opset_import: ["" : 17]>
            g (float[2,3,4] x) => (float[2,4] y)
            <int64[2] shape = {2,4}> {
                r = ReduceMean(x) <axes = [1], keepdims = 1>
                y = Reshape(r, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["ReduceMean"])
        feeds = {"x": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_generated_reduce_reshape_families(self):
        ops = (
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceMax",
            "ReduceMean",
            "ReduceMin",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
        )
        feeds = {"x": np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)}
        for op_type in ops:
            for axes_as_input in (True, False):
                opset = 18 if axes_as_input else (12 if op_type == "ReduceSum" else 17)
                if axes_as_input:
                    reduction = f"reduced = {op_type}(x, axes) <keepdims = 1>"
                    initializers = "<int64[1] axes = {1}, int64[2] shape = {2,4}>"
                else:
                    reduction = f"reduced = {op_type}(x) <axes = [1], keepdims = 1>"
                    initializers = "<int64[2] shape = {2,4}>"
                model_text = f"""
                    <ir_version: 10, opset_import: ["" : {opset}]>
                    g (float[2,3,4] x) => (float[2,4] y)
                    {initializers} {{
                        {reduction}
                        y = Reshape(reduced, shape)
                    }}
                """
                with self.subTest(op_type=op_type, axes_as_input=axes_as_input):
                    original, rewritten, count = _apply(model_text)
                    self.assertEqual(count, 1)
                    self.assertEqual([node.op_type for node in rewritten.graph], [op_type])
                    self.assert_equivalent(original, rewritten, feeds)

    def test_unsqueeze_reshape_zero_copy_special_case(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,1,3] x) => (float[2,1,3,1] y)
            <int64[1] axes = {2}, int64[4] shape = {0,1,-1,0}> {
                u = Unsqueeze(x, axes)
                y = Reshape(u, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Constant", "Unsqueeze"])
        feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 1, 3)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )

    def test_unsqueeze_reshape_special_case_rejects_ambiguous_empty_inference(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[N,1,3] x) => (float[N,1,3,1] y)
            <int64[1] axes = {2}, int64[4] shape = {0,1,-1,0}> {
                u = Unsqueeze(x, axes)
                y = Reshape(u, shape)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_transpose_unit_reshape_transpose_removes_data_transpose(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,1,3] x) => (float[2,3,1,1] y)
            <int64[4] shape = {3,2,1,1}> {
                t = Transpose(x) <perm = [2,0,1]>
                r = Reshape(t, shape)
                y = Transpose(r) <perm = [1,0,2,3]>
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Constant", "Reshape"])
        feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 1, 3)}
        np.testing.assert_array_equal(
            _evaluate(rewritten, feeds)[0], _evaluate(original, feeds)[0]
        )


if __name__ == "__main__":
    unittest.main()
