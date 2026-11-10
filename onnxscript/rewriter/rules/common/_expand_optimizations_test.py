# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.reference
import onnx_ir as ir

from onnxscript.rewriter.rules.common import _expand_optimizations as expand


def _evaluate(model: ir.Model, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    return onnx.reference.ReferenceEvaluator(ir.to_proto(model)).run(None, feeds)


def _apply(model_text: str) -> tuple[ir.Model, ir.Model, int]:
    original = ir.from_onnx_text(model_text)
    rewritten = ir.from_onnx_text(model_text)
    count = expand.rules.apply_to_model(rewritten)
    onnx.checker.check_model(ir.to_proto(rewritten), full_check=True)
    return original, rewritten, count


def _apply_rule(model_text: str, rule) -> tuple[ir.Model, ir.Model, int]:
    original = ir.from_onnx_text(model_text)
    rewritten = ir.from_onnx_text(model_text)
    count = rule.apply_to_model(rewritten)
    onnx.checker.check_model(ir.to_proto(rewritten), full_check=True)
    return original, rewritten, count


class ExpandOptimizationsTest(unittest.TestCase):
    def assert_equivalent(self, original, rewritten, feeds):
        expected = _evaluate(original, feeds)
        actual = _evaluate(rewritten, feeds)
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=0, atol=0, equal_nan=True)
            if np.issubdtype(got.dtype, np.floating):
                np.testing.assert_array_equal(np.signbit(got), np.signbit(want))

    def test_materialize_dynamic_expand_shape(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[N,1,3] x, int64[3] shape) => (float[N,4,3] y) {
                y = Expand(x, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Constant", "Expand"])
        feeds = {
            "x": np.arange(6, dtype=np.float32).reshape(2, 1, 3),
            "shape": np.array([2, 4, 3], dtype=np.int64),
        }
        self.assert_equivalent(original, rewritten, feeds)

    def test_all_one_expand_becomes_unsqueeze(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1] x) => (float[1,1,1] y)
            <int64[3] shape = {1,1,1}> {
                y = Expand(x, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Constant", "Unsqueeze"])
        self.assert_equivalent(original, rewritten, {"x": np.array([2.0], dtype=np.float32)})

    def test_move_elementwise_unary_before_expand(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,3] x) => (float[4,3] y)
            <int64[2] shape = {4,3}> {
                e = Expand(x, shape)
                y = Neg(e)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Neg", "Expand"])
        self.assert_equivalent(
            original, rewritten, {"x": np.arange(3, dtype=np.float32).reshape(1, 3)}
        )

    def test_move_unary_before_expand_preserves_special_values_exactly(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,3] x) => (float[2,3] y)
            <int64[2] shape = {2,3}> {
                e = Expand(x, shape)
                y = Neg(e)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        feeds = {"x": np.array([[-0.0, np.inf, np.nan]], dtype=np.float32)}
        expected = _evaluate(original, feeds)[0]
        actual = _evaluate(rewritten, feeds)[0]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))

    def test_move_unary_before_expand_rejects_new_empty_dimension(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1] x) => (float[0] y)
            <int64[1] shape = {0}> {
                e = Expand(x, shape)
                y = Reciprocal(e)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_generated_unary_rules_apply_with_dtype_correct_inputs(self):
        unary_ops = (
            "Abs",
            "Acos",
            "Acosh",
            "Asin",
            "Asinh",
            "Atan",
            "Atanh",
            "Ceil",
            "Cos",
            "Cosh",
            "Erf",
            "Exp",
            "Floor",
            "IsInf",
            "IsNaN",
            "Log",
            "Neg",
            "Not",
            "Reciprocal",
            "Relu",
            "Round",
            "Sigmoid",
            "Sign",
            "Sin",
            "Sinh",
            "Softplus",
            "Softsign",
            "Sqrt",
            "Tan",
            "Tanh",
        )
        domain_values = {
            "Acos": [0.25, 0.5, 0.75],
            "Acosh": [1.0, 1.5, 2.0],
            "Asin": [0.25, 0.5, 0.75],
            "Atanh": [0.25, 0.5, 0.75],
            "Log": [0.25, 0.5, 2.0],
            "Reciprocal": [0.25, 0.5, 2.0],
            "Sqrt": [0.25, 0.5, 2.0],
        }
        for op_type in unary_ops:
            if op_type == "Not":
                tensor_type = output_type = "bool"
                values = np.array([[True, False, True]])
            else:
                tensor_type = "float"
                output_type = "bool" if op_type in {"IsInf", "IsNaN"} else "float"
                values = np.array(
                    [domain_values.get(op_type, [-0.5, 0.25, 1.5])],
                    dtype=np.float32,
                )
            model_text = f"""
                <ir_version: 10, opset_import: ["" : 18]>
                g ({tensor_type}[1,3] x) => ({output_type}[2,3] y)
                <int64[2] shape = {{2,3}}> {{
                    expanded = Expand(x, shape)
                    y = {op_type}(expanded)
                }}
            """
            with self.subTest(op_type=op_type):
                original, rewritten, count = _apply(model_text)
                self.assertEqual(count, 1)
                self.assertEqual(
                    [node.op_type for node in rewritten.graph], [op_type, "Expand"]
                )
                self.assert_equivalent(original, rewritten, {"x": values})

    def test_move_cast_before_expand(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,3] x) => (int32[2,3] y)
            <int64[2] shape = {2,3}> {
                e = Expand(x, shape)
                y = Cast(e) <to = 6>
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assert_equivalent(
            original, rewritten, {"x": np.arange(3, dtype=np.float32).reshape(1, 3)}
        )

    def test_move_cast_preserves_additional_attributes(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 23]>
            g (float[1,3] x) => (int32[2,3] y)
            <int64[2] shape = {2,3}> {
                e = Expand(x, shape)
                y = Cast(e) <to = 6, saturate = 0>
            }
        """
        _, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        cast = next(iter(rewritten.graph))
        self.assertEqual(cast.op_type, "Cast")
        self.assertEqual(cast.attributes["saturate"].as_int(), 0)

    def test_swap_expand_unsqueeze_negative_axes(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,3] x) => (float[2,1,3,1] y)
            <int64[2] shape = {2,3}, int64[2] axes = {-3,-1}> {
                e = Expand(x, shape)
                y = Unsqueeze(e, axes)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual(
            [n.op_type for n in rewritten.graph], ["Unsqueeze", "Constant", "Expand"]
        )
        self.assert_equivalent(
            original, rewritten, {"x": np.arange(3, dtype=np.float32).reshape(1, 3)}
        )

    def test_swap_expand_unsqueeze_with_short_target(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3] x) => (float[2,1,3] y)
            <int64[1] shape = {3}, int64[1] axes = {1}> {
                e = Expand(x, shape)
                y = Unsqueeze(e, axes)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assert_equivalent(
            original, rewritten, {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        )

    def test_swap_expand_unsqueeze_rejects_rank_prepending_expand(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[3] x) => (float[2,1,3] y)
            <int64[2] shape = {2,3}, int64[1] axes = {1}> {
                e = Expand(x, shape)
                y = Unsqueeze(e, axes)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_fuse_expand_unsqueeze_expand(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,3] x) => (float[2,4,3] y)
            <int64[2] s1 = {2,3}, int64[1] axes = {1}, int64[3] s2 = {2,4,3}> {
                e = Expand(x, s1)
                u = Unsqueeze(e, axes)
                y = Expand(u, s2)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assert_equivalent(
            original, rewritten, {"x": np.arange(3, dtype=np.float32).reshape(1, 3)}
        )

    def test_fuse_expand_unsqueeze_expand_rule_directly(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,3] x) => (float[2,4,3] y)
            <int64[2] s1 = {2,3}, int64[1] axes = {1}, int64[3] s2 = {2,4,3}> {
                expanded = Expand(x, s1)
                unsqueezed = Unsqueeze(expanded, axes)
                y = Expand(unsqueezed, s2)
            }
        """
        original, rewritten, count = _apply_rule(
            model_text, expand.FuseExpandsAroundUnsqueeze.rule()
        )
        self.assertEqual(count, 1)
        self.assertEqual(
            [node.op_type for node in rewritten.graph],
            ["Unsqueeze", "Constant", "Expand"],
        )
        self.assert_equivalent(
            original, rewritten, {"x": np.arange(3, dtype=np.float32).reshape(1, 3)}
        )

    def test_fuse_expand_unsqueeze_expand_with_short_first_target(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[2,3] x) => (float[2,4,3] y)
            <int64[1] s1 = {3}, int64[1] axes = {1}, int64[3] s2 = {2,4,3}> {
                e = Expand(x, s1)
                u = Unsqueeze(e, axes)
                y = Expand(u, s2)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assert_equivalent(
            original, rewritten, {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        )

    def test_fuse_expand_unsqueeze_expand_rejects_rank_prepend(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[3] x) => (float[2,4,3] y)
            <int64[2] s1 = {2,3}, int64[1] axes = {1}, int64[3] s2 = {2,4,3}> {
                e = Expand(x, s1)
                u = Unsqueeze(e, axes)
                y = Expand(u, s2)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_unknown_symbolic_dimensions_do_not_prove_broadcast(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[?,1] x, float[?,1] z) => (float[4,3] y)
            <int64[2] shape = {4,3}> {
                a = Expand(x, shape)
                b = Expand(z, shape)
                y = Add(a, b)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_unit_expand_not_lowered_to_input_axes_before_opset13(self):
        model_text = """
            <ir_version: 7, opset_import: ["" : 12]>
            g (float[1] x) => (float[1,1,1] y)
            <int64[3] shape = {1,1,1}> {
                y = Expand(x, shape)
            }
        """
        rewritten = ir.from_onnx_text(model_text)
        count = expand.rules.apply_to_model(rewritten)
        self.assertEqual(count, 0)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Expand"])

    def test_unit_expand_uses_function_local_opset(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18, "local" : 1]>
            g (float[1] x) => (float[1,1,1] y) {
                y = local.expand(x)
            }
            <domain: "local", opset_import: ["" : 12]>
            expand (float[1] x) => (float[1,1,1] y) {
                shape = Constant <value = int64[3] {1,1,1}> ()
                y = Expand(x, shape)
            }
        """
        rewritten = ir.from_onnx_text(model_text)
        count = expand.rules.apply_to_model(rewritten)
        self.assertEqual(count, 0)
        function = rewritten.functions[("local", "expand", "")]
        self.assertEqual([node.op_type for node in function], ["Constant", "Expand"])

    def test_two_expands_become_one_after_binary(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,3] x, float[1,3] z) => (float[4,3] y)
            <int64[2] shape = {4,3}> {
                a = Expand(x, shape)
                b = Expand(z, shape)
                y = Add(a, b)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Add", "Expand"])
        feeds = {
            "x": np.arange(3, dtype=np.float32).reshape(1, 3),
            "z": np.ones((1, 3), dtype=np.float32),
        }
        self.assert_equivalent(original, rewritten, feeds)

    def test_two_expands_different_targets_rejected(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,3] x, float[1,3] z) => (float[4,3] y)
            <int64[2] s1 = {4,3}, int64[2] s2 = {1,3}> {
                a = Expand(x, s1)
                b = Expand(z, s2)
                y = Add(a, b)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_two_distinct_dynamic_expand_targets_are_not_assumed_equal(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,3] x, float[1,3] y, int64[2] shape1, int64[2] shape2)
                => (float[2,3] z) {
                left = Expand(x, shape1)
                right = Expand(y, shape2)
                z = Add(left, right)
            }
        """
        for shape1, shape2 in (
            (np.array([1, 3], dtype=np.int64), np.array([2, 3], dtype=np.int64)),
            (np.array([2, 3], dtype=np.int64), np.array([1, 3], dtype=np.int64)),
        ):
            with self.subTest(shape1=shape1.tolist(), shape2=shape2.tolist()):
                original, rewritten, count = _apply(model_text)
                self.assertEqual(count, 0)
                self.assertEqual(
                    [node.op_type for node in rewritten.graph],
                    ["Expand", "Expand", "Add"],
                )
                self.assert_equivalent(
                    original,
                    rewritten,
                    {
                        "x": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                        "y": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
                        "shape1": shape1,
                        "shape2": shape2,
                    },
                )

    def test_two_expands_binary_rejects_new_empty_dimension(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (int32[1] x, int32[1] z) => (int32[0] y)
            <int64[1] shape = {0}> {
                a = Expand(x, shape)
                b = Expand(z, shape)
                y = Div(a, b)
            }
        """
        _, _, count = _apply(model_text)
        self.assertEqual(count, 0)

    def test_generated_binary_rules_apply_with_dtype_correct_inputs(self):
        bool_ops = {"And", "Or", "Xor"}
        bitwise_ops = {"BitwiseAnd", "BitwiseOr", "BitwiseXor"}
        comparison_ops = {
            "Equal",
            "Greater",
            "GreaterOrEqual",
            "Less",
            "LessOrEqual",
        }
        ops = (
            "Add",
            "And",
            "BitwiseAnd",
            "BitwiseOr",
            "BitwiseXor",
            "Div",
            "Equal",
            "Greater",
            "GreaterOrEqual",
            "Less",
            "LessOrEqual",
            "Mod",
            "Mul",
            "Or",
            "Pow",
            "Sub",
            "Xor",
        )
        for op_type in ops:
            attributes = ""
            if op_type in bool_ops:
                tensor_type = output_type = "bool"
                x = np.array([[True, False, True]])
                y = np.array([[True, True, False]])
            elif op_type in bitwise_ops:
                tensor_type = output_type = "uint32"
                x = np.array([[1, 2, 4]], dtype=np.uint32)
                y = np.array([[1, 1, 2]], dtype=np.uint32)
            else:
                tensor_type = "float"
                output_type = "bool" if op_type in comparison_ops else "float"
                x = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
                y = np.array([[2.0, 1.0, 2.0]], dtype=np.float32)

            for family, left_expanded, right_expanded in (
                ("equal_expands", True, True),
                ("left_expand", True, False),
                ("right_expand", False, True),
            ):
                if left_expanded and right_expanded:
                    body = (
                        "left = Expand(x, shape)\n"
                        "                        right = Expand(y, shape)\n"
                        f"                        z = {op_type}(left, right){attributes}"
                    )
                elif left_expanded:
                    body = (
                        "expanded = Expand(x, shape)\n"
                        f"                        z = {op_type}(expanded, y){attributes}"
                    )
                else:
                    body = (
                        "expanded = Expand(y, shape)\n"
                        f"                        z = {op_type}(x, expanded){attributes}"
                    )
                model_text = f"""
                    <ir_version: 10, opset_import: ["" : 18]>
                    g ({tensor_type}[1,3] x, {tensor_type}[1,3] y)
                        => ({output_type}[2,3] z)
                    <int64[2] shape = {{2,3}}> {{
                        {body}
                    }}
                """
                with self.subTest(family=family, op_type=op_type):
                    original, rewritten, count = _apply(model_text)
                    self.assertEqual(count, 1)
                    self.assertEqual(
                        [node.op_type for node in rewritten.graph],
                        [op_type, "Expand"],
                    )
                    self.assert_equivalent(original, rewritten, {"x": x, "y": y})

    def test_binary_motion_preserves_bitshift_direction(self):
        for direction in ("LEFT", "RIGHT"):
            for family in ("equal_expands", "left_expand", "right_expand"):
                if family == "equal_expands":
                    body = (
                        "left = Expand(x, shape)\n"
                        "                        right = Expand(y, shape)\n"
                        f"                        z = BitShift(left, right) "
                        f'<direction = "{direction}">'
                    )
                elif family == "left_expand":
                    body = (
                        "expanded = Expand(x, shape)\n"
                        f"                        z = BitShift(expanded, y) "
                        f'<direction = "{direction}">'
                    )
                else:
                    body = (
                        "expanded = Expand(y, shape)\n"
                        f"                        z = BitShift(x, expanded) "
                        f'<direction = "{direction}">'
                    )
                model_text = f"""
                    <ir_version: 10, opset_import: ["" : 18]>
                    g (uint32[1,3] x, uint32[1,3] y) => (uint32[2,3] z)
                    <int64[2] shape = {{2,3}}> {{
                        {body}
                    }}
                """
                with self.subTest(direction=direction, family=family):
                    original, rewritten, count = _apply(model_text)
                    self.assertEqual(count, 1)
                    binary = next(iter(rewritten.graph))
                    self.assertEqual(binary.op_type, "BitShift")
                    self.assertEqual(binary.attributes["direction"].as_string(), direction)
                    self.assertEqual(expand.rules.apply_to_model(rewritten), 0)
                    onnx.checker.check_model(ir.to_proto(rewritten), full_check=True)
                    self.assert_equivalent(
                        original,
                        rewritten,
                        {
                            "x": np.array([[8, 4, 2]], dtype=np.uint32),
                            "y": np.array([[1, 2, 1]], dtype=np.uint32),
                        },
                    )

    def test_binary_motion_preserves_mod_fmod(self):
        for family in ("equal_expands", "left_expand", "right_expand"):
            if family == "equal_expands":
                body = (
                    "left = Expand(x, shape)\n"
                    "                        right = Expand(y, shape)\n"
                    "                        z = Mod(left, right) <fmod = 1>"
                )
            elif family == "left_expand":
                body = (
                    "expanded = Expand(x, shape)\n"
                    "                        z = Mod(expanded, y) <fmod = 1>"
                )
            else:
                body = (
                    "expanded = Expand(y, shape)\n"
                    "                        z = Mod(x, expanded) <fmod = 1>"
                )
            model_text = f"""
                <ir_version: 10, opset_import: ["" : 18]>
                g (float[1,3] x, float[1,3] y) => (float[2,3] z)
                <int64[2] shape = {{2,3}}> {{
                    {body}
                }}
            """
            with self.subTest(family=family):
                original, rewritten, count = _apply(model_text)
                self.assertEqual(count, 1)
                binary = next(iter(rewritten.graph))
                self.assertEqual(binary.op_type, "Mod")
                self.assertEqual(binary.attributes["fmod"].as_int(), 1)
                self.assertEqual(expand.rules.apply_to_model(rewritten), 0)
                onnx.checker.check_model(ir.to_proto(rewritten), full_check=True)
                self.assert_equivalent(
                    original,
                    rewritten,
                    {
                        "x": np.array([[-5.5, 5.5, -5.5]], dtype=np.float32),
                        "y": np.array([[2.0, -2.0, -2.0]], dtype=np.float32),
                    },
                )

    def test_binary_motion_rejects_reference_attributes(self):
        model = ir.from_onnx_text(
            """
            <ir_version: 10, opset_import: ["" : 18, "local" : 1]>
            graph (uint32[1,3] x, uint32[1,3] y) => (uint32[2,3] z) {
                z = local.shift <direction = "LEFT"> (x, y)
            }
            <domain: "local", opset_import: ["" : 18]>
            shift <direction> (uint32[1,3] x, uint32[1,3] y) => (uint32[2,3] z) {
                shape = Constant <value = int64[2] {2,3}> ()
                left = Expand(x, shape)
                right = Expand(y, shape)
                z = BitShift <direction: string = @direction> (left, right)
            }
            """
        )
        self.assertEqual(expand.rules.apply_to_model(model), 0)
        function = model.functions[("local", "shift", "")]
        self.assertTrue(function[-1].attributes["direction"].is_ref())

    def test_one_expand_moves_after_binary(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,3] x, float[1,3] z) => (float[4,3] y)
            <int64[2] shape = {4,3}> {
                a = Expand(x, shape)
                y = Sub(a, z)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Sub", "Expand"])
        feeds = {
            "x": np.arange(3, dtype=np.float32).reshape(1, 3),
            "z": np.ones((1, 3), dtype=np.float32),
        }
        self.assert_equivalent(original, rewritten, feeds)

    def test_swap_expand_reshape_rank_three_special_case(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,1,1] x) => (float[4,1,1] y)
            <int64[3] es = {4,1,1}, int64[3] rs = {0,1,-1}> {
                e = Expand(x, es)
                y = Reshape(e, rs)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Reshape", "Expand"])
        self.assert_equivalent(
            original, rewritten, {"x": np.ones((1, 1, 1), dtype=np.float32)}
        )

    def test_expand_cast_where_moves_before_expand(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 23]>
            g (float[1,3] x, float[1,3] other) => (float[4,3] y)
            <int64[2] shape = {4,3}> {
                e = Expand(x, shape)
                c = Cast(e) <to = 9, saturate = 0>
                y = Where(c, e, other)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Cast", "Where", "Expand"])
        self.assertEqual(next(iter(rewritten.graph)).attributes["saturate"].as_int(), 0)
        feeds = {
            "x": np.array([[0.0, 2.0, -1.0]], dtype=np.float32),
            "other": np.array([[5.0, 6.0, 7.0]], dtype=np.float32),
        }
        self.assert_equivalent(original, rewritten, feeds)

    def test_expand_cast_where_else_branch_moves_before_expand(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[1,3] x, float[1,3] other) => (float[2,3] y)
            <int64[2] shape = {2,3}> {
                expanded = Expand(x, shape)
                condition = Cast(expanded) <to = 9>
                y = Where(condition, other, expanded)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Cast", "Where", "Expand"])
        feeds = {
            "x": np.array([[0.0, 2.0, -1.0]], dtype=np.float32),
            "other": np.array([[5.0, 6.0, 7.0]], dtype=np.float32),
        }
        self.assert_equivalent(original, rewritten, feeds)

    def test_concat_expand_replaces_unchanged_symbolic_dimensions(self):
        model_text = """
            <ir_version: 10, opset_import: ["" : 18]>
            g (float[N,1,M] x, int64[1] n, int64[1] middle, int64[1] m)
                => (float[N,4,M] y) {
                shape = Concat(n, middle, m) <axis = 0>
                y = Expand(x, shape)
            }
        """
        original, rewritten, count = _apply(model_text)
        self.assertEqual(count, 1)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Constant", "Expand"])
        feeds = {
            "x": np.arange(6, dtype=np.float32).reshape(2, 1, 3),
            "n": np.array([2], dtype=np.int64),
            "middle": np.array([4], dtype=np.int64),
            "m": np.array([3], dtype=np.int64),
        }
        self.assert_equivalent(original, rewritten, feeds)

    def test_concat_expand_rank_variants_are_idempotent(self):
        for rank in (2, 3, 4, 5):
            input_shape = [2, 1] + [3] * (rank - 2)
            output_shape = ["2", "K"] + ["3"] * (rank - 2)
            input_type = ",".join(map(str, input_shape))
            output_type = ",".join(output_shape)
            arguments = ", ".join(f"int64[1] p{i}" for i in range(rank))
            concat_args = ", ".join(f"p{i}" for i in range(rank))
            model_text = f"""
                <ir_version: 10, opset_import: ["" : 18]>
                g (float[{input_type}] x, {arguments}) => (float[{output_type}] y) {{
                    shape = Concat({concat_args}) <axis = 0>
                    y = Expand(x, shape)
                }}
            """
            concrete_output_shape = [2, 4] + [3] * (rank - 2)
            feeds = {
                "x": np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape),
                **{
                    f"p{i}": np.array([dim], dtype=np.int64)
                    for i, dim in enumerate(concrete_output_shape)
                },
            }
            with self.subTest(rank=rank):
                original, rewritten, count = _apply(model_text)
                self.assertEqual(count, 1)
                self.assertEqual(
                    [node.op_type for node in rewritten.graph],
                    ["Constant", "Concat", "Expand"],
                )
                concat = list(rewritten.graph)[1]
                unit_inputs = [
                    value for index, value in enumerate(concat.inputs) if index != 1
                ]
                self.assertEqual(unit_inputs[0].producer().op_type, "Constant")
                self.assertTrue(all(value is unit_inputs[0] for value in unit_inputs[1:]))
                self.assertEqual(expand.rules.apply_to_model(rewritten), 0)
                onnx.checker.check_model(ir.to_proto(rewritten), full_check=True)
                self.assert_equivalent(original, rewritten, feeds)


if __name__ == "__main__":
    unittest.main()
