# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np

from onnxscript import ir
from onnxscript.ir.passes.common import _c_api_utils, shape_inference


class TestShapeInferencePass(unittest.TestCase):
    def test_pass_is_in_place(self):
        self.assertTrue(shape_inference.ShapeInferencePass().in_place)

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

        tape = ir.tape.Tape()

        output = tape.op("Add", inputs=inputs)

        model = ir.Model(
            ir.Graph(
                inputs=inputs,
                outputs=[output],
                nodes=tape.nodes,
                opset_imports={"": 20},
            ),
            ir_version=10,
        )
        self.assertIsNone(output.shape)
        self.assertIsNone(output.dtype)

        # Perform shape inference
        result = shape_inference.ShapeInferencePass()(model)
        self.assertTrue(result.modified)
        self.assertEqual(result.model.graph.node(0).outputs[0].shape, ir.Shape((1, 2)))
        self.assertEqual(result.model.graph.node(0).outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(result.model.graph.outputs[0].shape, ir.Shape((1, 2)))
        self.assertEqual(result.model.graph.outputs[0].dtype, ir.DataType.FLOAT)

    def test_pass_with_initializers(self):
        # _BIG_TENSOR_SIZE_LIMIT is in bytes, but we create big_dim as size
        # of a tensor. This is fine as we just need to create a big tensor whose size
        # passes _BIG_TENSOR_SIZE_LIMIT
        big_dim = _c_api_utils._BIG_TENSOR_SIZE_LIMIT * 2  # pylint: disable=protected-access
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

        tape = ir.tape.Tape()

        # Shape and type are not explicitly set for the initializer but it should still work
        initializer = ir.Value(
            name="initializer", const_value=ir.tensor([[2, 3]], dtype=ir.DataType.FLOAT)
        )
        val_add = tape.op("Add", inputs=inputs)
        val_mul = tape.op("Mul", inputs=[val_add, initializer])

        model = ir.Model(
            ir.Graph(
                inputs=inputs,
                outputs=[val_mul],
                nodes=tape.nodes,
                opset_imports={"": 20},
                initializers=[inputs[1], initializer],
            ),
            ir_version=10,
        )

        self.assertIsNone(val_add.shape)
        self.assertIsNone(val_add.dtype)
        self.assertIsNone(val_mul.shape)
        self.assertIsNone(val_mul.dtype)
        self.assertIsNone(initializer.shape)
        self.assertIsNone(initializer.dtype)

        # Perform shape inference
        result = shape_inference.ShapeInferencePass()(model)
        self.assertTrue(result.modified)
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


if __name__ == "__main__":
    unittest.main()
