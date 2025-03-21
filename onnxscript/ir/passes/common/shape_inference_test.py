# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

from onnxscript import ir
from onnxscript.ir.passes.common import shape_inference


class TestShapeInference(unittest.TestCase):
    def test_shape_inference(self):
        # Create a simple ONNX model with shape inference
        # Define the model
        inputs = [
            ir.Value(
                name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2))
            ),
            ir.Value(
                name="input_b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2))
            ),
        ]

        add_node = ir.Node("", "Add", inputs=inputs)

        model = ir.Model(
            ir.Graph(
                inputs=inputs,
                outputs=add_node.outputs,
                nodes=[add_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )
        self.assertIsNone(add_node.outputs[0].shape)
        self.assertIsNone(add_node.outputs[0].dtype)

        # Perform shape inference
        result = shape_inference.ShapeInferencePass()(model)
        self.assertFalse(result.modified)
        self.assertEqual(result.model.graph.node(0).outputs[0].shape, ir.Shape((1, 2)))
        self.assertEqual(result.model.graph.node(0).outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(result.model.graph.outputs[0].shape, ir.Shape((1, 2)))
        self.assertEqual(result.model.graph.outputs[0].dtype, ir.DataType.FLOAT)


if __name__ == "__main__":
    unittest.main()
