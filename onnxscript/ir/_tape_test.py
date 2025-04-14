# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

from onnxscript import ir


class TestTape(unittest.TestCase):
    def test_op(self):
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

        _ = tape.op("Add", inputs=inputs)

        self.assertEqual([n.op_type for n in tape.nodes], ["Add"])

    def test_initializers(self):
        inputs = [
            ir.Value(
                name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2))
            ),
            ir.Value(
                name="input_b",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape((2, 1)),
                const_value=ir.tensor([[42]] * 2, dtype=ir.DataType.FLOAT),
            ),
        ]

        tape = ir.tape.Tape()

        # Shape and type are not explicitly set for the initializer but it should still work
        initializer = tape.initializer(
            ir.tensor([[2, 3]], dtype=ir.DataType.FLOAT), name="initializer"
        )
        val_add = tape.op("Add", inputs=inputs)
        _ = tape.op("Mul", inputs=[val_add, initializer])

        self.assertEqual([n.op_type for n in tape.nodes], ["Add", "Mul"])
        self.assertEqual(tape.initializers, (initializer,))

    def test_op_multi_out(self):
        inputs = [
            ir.Value(
                name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2))
            ),
            ir.Value(
                name="input_b",
                type=ir.TensorType(ir.DataType.FLOAT),
                shape=ir.Shape((2, 1)),
                const_value=ir.tensor([[42]] * 2, dtype=ir.DataType.FLOAT),
            ),
        ]

        tape = ir.tape.Tape()

        out1, out2, out3 = tape.op_multi_output("SomeOp", inputs=inputs, num_outputs=3)  # pylint: disable=unbalanced-tuple-unpacking
        _ = tape.op("SomeOtherOp", inputs=[out1, out2, out3])

        self.assertEqual([n.op_type for n in tape.nodes], ["SomeOp", "SomeOtherOp"])


if __name__ == "__main__":
    unittest.main()
