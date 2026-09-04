# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np

from onnxscript import ir
from onnxscript.rewriter import testing
from onnxscript.rewriter.rules.common import _materialize_reshape_shape


class MaterializeReshapeShapeTest(unittest.TestCase):
    def test_fully_static_output_shape_materializes(self):
        """When output shape is fully static, replace dynamic shape input with constant."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data) => (float[2, 3] output)
            {
                shape = Shape(data)
                output = Reshape(data, shape)
            }
        """
        )
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = ir.Shape([2, 3])
                break
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        reshape_nodes = [n for n in model.graph if n.op_type == "Reshape"]
        self.assertEqual(len(reshape_nodes), 1)
        shape_input = reshape_nodes[0].inputs[1]
        self.assertIsNotNone(shape_input.const_value)
        self.assertEqual(shape_input.const_value.numpy().tolist(), [2, 3])

    def test_one_symbolic_dim_uses_minus_one(self):
        """When output has one symbolic dim, replace it with -1."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data) => (float[B, 3] output)
            {
                shape = Shape(data)
                output = Reshape(data, shape)
            }
        """
        )
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = ir.Shape(["B", 3])
                break
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        reshape_nodes = [n for n in model.graph if n.op_type == "Reshape"]
        self.assertEqual(len(reshape_nodes), 1)
        shape_input = reshape_nodes[0].inputs[1]
        self.assertIsNotNone(shape_input.const_value)
        self.assertEqual(shape_input.const_value.numpy().tolist(), [-1, 3])

    def test_two_symbolic_dims_not_materialized(self):
        """When output has two symbolic dims, the rule should not fire."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data) => (float[B, C] output)
            {
                shape = Shape(data)
                output = Reshape(data, shape)
            }
        """
        )
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = ir.Shape(["B", "C"])
                break
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_constant_shape_input_not_replaced(self):
        """When the shape input is already a constant, the rule should not fire."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data) => (float[2, 3] output)
            {
                shape = Constant<value: tensor = int64[2] {2, 3}>()
                output = Reshape(data, shape)
            }
        """
        )
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_unknown_output_shape_not_materialized(self):
        """When the output shape is unknown, the rule should not fire."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data) => (float output)
            {
                shape = Shape(data)
                output = Reshape(data, shape)
            }
        """
        )
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = None
                break
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 0)

    def test_allowzero_attribute_preserved(self):
        """The allowzero attribute should be preserved on the new Reshape."""
        model = ir.from_onnx_text(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[6] data) => (float[2, 3] output)
            {
                shape = Shape(data)
                output = Reshape<allowzero=1>(data, shape)
            }
        """
        )
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = ir.Shape([2, 3])
                break
        count = _materialize_reshape_shape.rules.apply_to_model(model)
        self.assertEqual(count, 1)
        reshape_nodes = [n for n in model.graph if n.op_type == "Reshape"]
        self.assertEqual(len(reshape_nodes), 1)
        allowzero = reshape_nodes[0].attributes.get_int("allowzero", 0)
        self.assertEqual(allowzero, 1)

    def test_numerical_correctness_static(self):
        """Verify numerical equivalence for fully static materialization."""
        # Build a model where a dynamic Concat produces the shape for Reshape.
        # After materialization, the Reshape uses a constant shape.
        model_text = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[12] data, float[3, 4] ref) => (float[3, 4] output)
            {
                shape = Shape(ref)
                output = Reshape(data, shape)
            }
        """
        original = ir.from_onnx_text(model_text)
        model = ir.from_onnx_text(model_text)
        for node in model.graph:
            if node.op_type == "Reshape":
                node.outputs[0].shape = ir.Shape([3, 4])
                break
        _materialize_reshape_shape.rules.apply_to_model(model)
        testing.assert_numerically_equal(
            original,
            model,
            (
                np.arange(12).astype(np.float32),
                np.zeros((3, 4), dtype=np.float32),
            ),
        )


if __name__ == "__main__":
    unittest.main()
