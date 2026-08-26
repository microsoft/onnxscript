# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx.parser

from onnxscript import ir
from onnxscript.rewriter import testing
from onnxscript.rewriter.rules.common import _collapse_slices

_INT64_MAX = 9223372036854775807


class TwoReshapesMatMulReshapeTest(unittest.TestCase):
    def test_slice_is_redundant_when_ends_is_greater_than_input_shape(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[512, 16, 112] data) => (float[512, 16, 112] output)
            {
                starts = Constant<value: tensor = int64[1] {0}>()
                ends = Constant<value: tensor = int64[1] {9999}>()
                axes = Constant<value: tensor = int64[1] {0}>()
                steps = Constant<value: tensor = int64[1] {1}>()
                output = Slice (data, starts, ends, axes, steps)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = _collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 5)
        self.assertIn("Identity", [node.op_type for node in model.graph])
        testing.assert_numerically_equal(
            model_proto,
            model,
            (np.random.rand(512, 16, 112).astype(np.float32),),
        )

    def test_slice_is_redundant_when_ends_reaches_int64_max(self):
        model_proto = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[512, 16, 112] data) => (float[512, 16, 112] output)
            {{
                starts = Constant<value: tensor = int64[1] {{0}}>()
                ends = Constant<value: tensor = int64[1] {{{_INT64_MAX}}}>()
                axes = Constant<value: tensor = int64[1] {{0}}>()
                steps = Constant<value: tensor = int64[1] {{1}}>()
                output = Slice (data, starts, ends, axes, steps)
            }}
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = _collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 5)
        self.assertIn("Identity", [node.op_type for node in model.graph])
        testing.assert_numerically_equal(
            model_proto,
            model,
            (np.random.rand(512, 16, 112).astype(np.float32),),
        )

    def test_slice_unequal_dynamic_shape(self):
        model_proto = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[L, M, N] data) => (float[P, M, N] output)
            {{
                starts = Constant<value: tensor = int64[1] {{0}}>()
                ends = Constant<value: tensor = int64[1] {{{9}}}>()
                axes = Constant<value: tensor = int64[1] {{0}}>()
                steps = Constant<value: tensor = int64[1] {{1}}>()
                output = Slice (data, starts, ends, axes, steps)
            }}
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = _collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_slice_equal_dynamic_shape(self):
        model_proto = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[L, M, N] data) => (float[L, M, N] output)
            {{
                starts = Constant<value: tensor = int64[1] {{0}}>()
                ends = Constant<value: tensor = int64[1] {{{9}}}>()
                axes = Constant<value: tensor = int64[1] {{0}}>()
                steps = Constant<value: tensor = int64[1] {{1}}>()
                output = Slice (data, starts, ends, axes, steps)
            }}
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = _collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 1)

    def test_slice_equal_dynamic_shape_but_step_reverse(self):
        model_proto = onnx.parser.parse_model(
            f"""
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[L, M, N] data) => (float[L, M, N] output)
            {{
                starts = Constant<value: tensor = int64[1] {{0}}>()
                ends = Constant<value: tensor = int64[1] {{{9}}}>()
                axes = Constant<value: tensor = int64[1] {{0}}>()
                steps = Constant<value: tensor = int64[1] {{-1}}>()
                output = Slice (data, starts, ends, axes, steps)
            }}
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = _collapse_slices.rules.apply_to_model(model)
        # Should not change the output shape if we did not use the default step of 1
        self.assertEqual(count, 0)

    def test_multi_element_steps_all_ones_collapses(self):
        """Slice with multi-axis steps=[1,1] and matching shapes should collapse."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[L, M] data) => (float[L, M] output)
            {
                starts = Constant<value: tensor = int64[2] {0, 0}>()
                ends = Constant<value: tensor = int64[2] {9999, 9999}>()
                axes = Constant<value: tensor = int64[2] {0, 1}>()
                steps = Constant<value: tensor = int64[2] {1, 1}>()
                output = Slice (data, starts, ends, axes, steps)
            }
        """
        )
        count = _collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertIn("Identity", [node.op_type for node in model.graph])

    def test_multi_element_steps_with_non_one_does_not_collapse(self):
        """Slice with steps containing a non-1 element should not collapse."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[10, 20] data) => (float[10, 10] output)
            {
                starts = Constant<value: tensor = int64[2] {0, 0}>()
                ends = Constant<value: tensor = int64[2] {10, 20}>()
                axes = Constant<value: tensor = int64[2] {0, 1}>()
                steps = Constant<value: tensor = int64[2] {1, 2}>()
                output = Slice (data, starts, ends, axes, steps)
            }
        """
        )
        count = _collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_multi_element_steps_numerical_correctness(self):
        """Verify numerical correctness of multi-axis collapse."""
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[4, 5] data) => (float[4, 5] output)
            {
                starts = Constant<value: tensor = int64[2] {0, 0}>()
                ends = Constant<value: tensor = int64[2] {100, 100}>()
                axes = Constant<value: tensor = int64[2] {0, 1}>()
                steps = Constant<value: tensor = int64[2] {1, 1}>()
                output = Slice (data, starts, ends, axes, steps)
            }
        """
        original = ir.from_onnx_text(model_text)
        model = ir.from_onnx_text(model_text)
        _collapse_slices.rules.apply_to_model(model)
        testing.assert_numerically_equal(
            original,
            model,
            (np.random.rand(4, 5).astype(np.float32),),
        )
