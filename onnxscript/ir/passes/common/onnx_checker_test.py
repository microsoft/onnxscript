# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

from onnxscript import ir
from onnxscript.ir.passes.common import onnx_checker


class TestCheckerPass(unittest.TestCase):
    def test_pass_is_no_op(self):
        checker_pass = onnx_checker.CheckerPass()
        self.assertTrue(checker_pass.in_place)
        self.assertFalse(checker_pass.changes_input)

    def test_check_simple_model(self):
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
        output.shape = ir.Shape((1, 2))
        output.dtype = ir.DataType.FLOAT

        model = ir.Model(
            ir.Graph(
                inputs=inputs,
                outputs=[output],
                nodes=tape.nodes,
                opset_imports={"": 20},
                name="test_model",
            ),
            ir_version=10,
        )
        # No exception should be raised
        onnx_checker.CheckerPass()(model)

    def test_check_invalid_model(self):
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
        output.shape = ir.Shape((1, 2))
        output.dtype = ir.DataType.FLOAT

        model = ir.Model(
            ir.Graph(
                inputs=inputs,
                outputs=[output],
                nodes=tape.nodes,
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        with self.assertRaisesRegex(
            Exception, "Field 'name' of 'graph' is required to be non-empty"
        ):
            onnx_checker.CheckerPass()(model)


if __name__ == "__main__":
    unittest.main()
