# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx.parser
import onnx.shape_inference

from onnxscript import ir
from onnxscript.rewriter import collapse_slices, testing

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
        count = collapse_slices.rules.apply_to_model(model)
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
        count = collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        self.assertEqual(len(model.graph), 5)
        self.assertIn("Identity", [node.op_type for node in model.graph])
        testing.assert_numerically_equal(
            model_proto,
            model,
            (np.random.rand(512, 16, 112).astype(np.float32),),
        )

    def test_slice_pattern_is_not_matched_when_input_is_dynamic(self):
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
        count = collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 0)
