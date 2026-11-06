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


class ExpandOptimizationsTest(unittest.TestCase):
    def assert_equivalent(self, original, rewritten, feeds):
        expected = _evaluate(original, feeds)
        actual = _evaluate(rewritten, feeds)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=0, atol=0)

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
        _, rewritten, count = _apply(model_text)
        self.assertEqual(count, 0)
        self.assertEqual([n.op_type for n in rewritten.graph], ["Expand"])

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


if __name__ == "__main__":
    unittest.main()
