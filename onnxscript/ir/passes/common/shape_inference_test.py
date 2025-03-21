# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np

from onnxscript import ir
from onnxscript.ir.passes.common import shape_inference


class TestShapeInferencePass(unittest.TestCase):
    def test_pass(self):
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

    def test_pass_with_initializers(self):
        big_dim = shape_inference._BIG_TENSOR_SIZE_LIMIT * 2
        inputs = [
            ir.Value(
                name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2))
            ),
            ir.Value(
                name="input_b",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape((big_dim, 1)),
                const_value=ir.tensor([[42]] * big_dim, dtype=ir.DataType.FLOAT),
            ),
        ]

        # Shape and type are not explicitly set for the initializer but it should still work
        initializer = ir.Value(
            name="initializer", const_value=ir.tensor([[2, 3]], dtype=ir.DataType.FLOAT)
        )

        add_node = ir.Node("", "Add", inputs=[*inputs])
        mul_node = ir.Node("", "Mul", inputs=[add_node.outputs[0], initializer])

        model = ir.Model(
            graph := ir.Graph(
                inputs=inputs,
                outputs=mul_node.outputs,
                nodes=[add_node, mul_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )
        graph.register_initializer(inputs[1])
        graph.register_initializer(initializer)

        self.assertIsNone(add_node.outputs[0].shape)
        self.assertIsNone(add_node.outputs[0].dtype)
        self.assertIsNone(mul_node.outputs[0].shape)
        self.assertIsNone(mul_node.outputs[0].dtype)
        self.assertIsNone(initializer.shape)
        self.assertIsNone(initializer.dtype)

        # Perform shape inference
        result = shape_inference.ShapeInferencePass()(model)
        self.assertFalse(result.modified)
        self.assertEqual(result.model.graph.node(0).outputs[0].shape, ir.Shape((big_dim, 2)))
        self.assertEqual(result.model.graph.node(0).outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(result.model.graph.node(1).outputs[0].shape, ir.Shape((big_dim, 2)))
        self.assertEqual(result.model.graph.node(1).outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(
            result.model.graph.initializers["initializer"].shape, ir.Shape((1, 2))
        )
        self.assertEqual(
            result.model.graph.initializers["initializer"].dtype, ir.DataType.FLOAT
        )
        self.assertEqual(result.model.graph.outputs[0].shape, ir.Shape((big_dim, 2)))
        self.assertEqual(result.model.graph.outputs[0].dtype, ir.DataType.FLOAT)

        # Check that the initializer correctly appears in the result
        self.assertEqual(len(result.model.graph.inputs), 2)
        self.assertEqual(len(result.model.graph.initializers), 2)
        np.testing.assert_array_equal(
            result.model.graph.initializers["input_b"].const_value.numpy(),
            np.array([[42]] * big_dim, dtype=np.float32),
            strict=True,
        )
        self.assertEqual(
            result.model.graph.initializers["input_b"].const_value.dtype,
            ir.DataType.FLOAT,
        )
        np.testing.assert_array_equal(
            result.model.graph.initializers["initializer"].const_value.numpy(),
            np.array([[2.0, 3.0]], dtype=np.float32),
            strict=True,
        )
        self.assertEqual(
            result.model.graph.initializers["initializer"].const_value.dtype,
            ir.DataType.FLOAT,
        )

        # Check that the original model is not modified
        self.assertIsNone(add_node.outputs[0].shape)
        self.assertIsNone(add_node.outputs[0].dtype)
        self.assertIsNone(mul_node.outputs[0].shape)
        self.assertIsNone(mul_node.outputs[0].dtype)
        self.assertEqual(len(model.graph.inputs), 2)
        self.assertEqual(len(model.graph.initializers), 2)
        self.assertIs(model.graph.initializers["input_b"].const_value, inputs[1].const_value)
        self.assertEqual(len(model.graph.outputs), 1)
        self.assertEqual(model.graph.outputs[0].shape, None)
        self.assertEqual(model.graph.outputs[0].dtype, None)
        # Check that the initializer is not modified
        self.assertIs(
            model.graph.initializers["initializer"].const_value, initializer.const_value
        )


if __name__ == "__main__":
    unittest.main()
