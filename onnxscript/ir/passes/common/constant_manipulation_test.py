# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import parameterized

from onnxscript import ir
from onnxscript.ir.passes.common import constant_manipulation


class TestLiftConstantsToInitializersPass(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (ir.DataType.FLOAT, np.float32),
            (ir.DataType.INT64, np.int64),
        ]
    )
    def test_pass_with_lifting_constants_to_initializers(self, ir_dtype, numpy_dtype):
        inputs = [
            ir.Value(name="input_a", type=ir.TensorType(ir_dtype), shape=ir.Shape((2, 3))),
            ir.Value(
                name="input_b",
                type=ir.TensorType(ir_dtype),
                shape=ir.Shape((2, 3)),
            ),
        ]

        constant_tensor = ir.tensor(np.random.rand(2, 3).astype(numpy_dtype))
        attribute = ir.convenience.convert_attributes({"value": constant_tensor})
        const_node = ir.Node("", "Constant", inputs=[], attributes=attribute, num_outputs=1)
        add_node = ir.Node("", "Add", inputs=[inputs[0], const_node.outputs[0]])
        mul_node = ir.Node("", "Mul", inputs=[add_node.outputs[0], inputs[1]])

        model = ir.Model(
            graph=ir.Graph(
                inputs=inputs,
                outputs=mul_node.outputs,
                nodes=[const_node, add_node, mul_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Check that the initializer is not in the graph yet
        self.assertEqual(len(model.graph.initializers), 0)
        # And 1 constant node
        self.assertEqual(len([node for node in model.graph if node.op_type == "Constant"]), 1)

        # Perform lift constants to initializers
        result = constant_manipulation.LiftConstantsToInitializersPass()(model)
        self.assertTrue(result.modified)
        # Check that the constant node is lifted to an initializer
        self.assertEqual(len(result.model.graph.initializers), 1)
        # Check the value
        self.assertEqual(
            result.model.graph.initializers[
                "val_0"
            ].const_value,  # name created by name_authority
            constant_tensor,
        )
        # And 0 constant node
        self.assertEqual(
            len([node for node in result.model.graph if node.op_type == "Constant"]), 0
        )
