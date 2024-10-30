# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx.parser
import onnx.shape_inference

from onnxscript import ir
from onnxscript.rewriter import collapse_slices


def _infer_shapes(model: ir.Model) -> ir.Model:
    """Run shape inference on the IR model."""
    # TODO: Update when shape inference is supported on the IR
    return ir.serde.deserialize_model(
        onnx.shape_inference.infer_shapes(ir.serde.serialize_model(model))
    )


class TwoReshapesMatMulReshapeTest(unittest.TestCase):
    def test_two_consecutive_slices_with_only_axes_difference_should_be_collpased_to_one(self):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[512, 16, 112, 112] data_a) => (float[512, 16, 112, 112] output)
            {
                starts_a = Constant<value: tensor = int64[1] {0}>()
                starts_b = Constant<value: tensor = int64[1] {0}>()
                ends_a = Constant<value: tensor = int64[1] {999}>()
                ends_b = Constant<value: tensor = int64[1] {999}>()
                axes_a = Constant<value: tensor = int64[1] {0}>()
                axes_b = Constant<value: tensor = int64[1] {2}>()
                steps_a = Constant<value: tensor = int64[1] {1}>()
                steps_b = Constant<value: tensor = int64[1] {1}>()
                intermediate = Slice (data_a, starts_a, ends_a, axes_a, steps_a)
                output = Slice (intermediate, starts_b, ends_b, axes_b, steps_b)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        _infer_shapes(model)

    def test_three_consecutive_slices_with_only_axes_difference_should_be_collpased_to_one(
        self,
    ):
        model_proto = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[512, 16, 112, 112] data_a) => (float[512, 16, 112, 112] output)
            {
                starts_a = Constant<value: tensor = int64[1] {0}>()
                starts_b = Constant<value: tensor = int64[1] {0}>()
                starts_c = Constant<value: tensor = int64[1] {0}>()
                ends_a = Constant<value: tensor = int64[1] {999}>()
                ends_b = Constant<value: tensor = int64[1] {999}>()
                ends_c = Constant<value: tensor = int64[1] {999}>()
                axes_a = Constant<value: tensor = int64[1] {0}>()
                axes_b = Constant<value: tensor = int64[1] {2}>()
                axes_c = Constant<value: tensor = int64[1] {3}>()
                steps_a = Constant<value: tensor = int64[1] {1}>()
                steps_b = Constant<value: tensor = int64[1] {1}>()
                steps_c = Constant<value: tensor = int64[1] {1}>()
                intermediate_1 = Slice (data_a, starts_a, ends_a, axes_a, steps_a)
                intermediate_2 = Slice (intermediate_1, starts_b, ends_b, axes_b, steps_b)
                output = Slice (intermediate_2, starts_c, ends_c, axes_c, steps_c)
            }
        """
        )
        model = ir.serde.deserialize_model(model_proto)
        count = collapse_slices.rules.apply_to_model(model)
        self.assertEqual(count, 2)
        _infer_shapes(model)
