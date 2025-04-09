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

    def test_pass_with_lifting_constants_to_initializers_within_subgraph(self):
        input_value = ir.Value(
            name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))
        )

        then_constant_tensor = ir.tensor(np.random.rand(2, 3).astype(np.float32))
        attribute = ir.convenience.convert_attributes({"value": then_constant_tensor})
        then_const_node = ir.Node(
            "", "Constant", inputs=[], attributes=attribute, num_outputs=1
        )
        # then branch adds the constant to the input
        # else branch multiplies the input by the constant
        add_node = ir.Node("", "Add", inputs=[input_value, then_const_node.outputs[0]])
        then_graph = ir.Graph(
            inputs=[input_value],
            outputs=[add_node.outputs[0]],
            nodes=[then_const_node, add_node],
            opset_imports={"": 20},
        )
        else_constant_tensor = ir.tensor(np.random.rand(2, 3).astype(np.float32))
        attribute = ir.convenience.convert_attributes({"value": else_constant_tensor})
        else_const_node = ir.Node(
            "", "Constant", inputs=[], attributes=attribute, num_outputs=1
        )
        mul_node = ir.Node("", "Mul", inputs=[input_value, else_const_node.outputs[0]])
        else_graph = ir.Graph(
            inputs=[input_value],
            outputs=[mul_node.outputs[0]],
            nodes=[else_const_node, mul_node],
            opset_imports={"": 20},
        )
        # create a conditional node that uses the then and else graphs
        attribute = ir.convenience.convert_attributes(
            {"then_branch": then_graph, "else_branch": else_graph}
        )
        cond_node = ir.Node(
            "",
            "If",
            inputs=[input_value],
            attributes=attribute,
            num_outputs=1,
        )
        # construnct the model
        main_graph = ir.Graph(
            inputs=[input_value],
            outputs=cond_node.outputs,
            nodes=[cond_node],
            opset_imports={"": 20},
        )
        main_graph.sort()
        model = ir.Model(
            graph=main_graph,
            ir_version=10,
        )
        result = constant_manipulation.LiftConstantsToInitializersPass()(model)
        self.assertTrue(result.modified)
        # Check that the constant node is lifted to the subgraph initializers
        for node in ir.traversal.RecursiveGraphIterator(result.model.graph):
            if node.op_type == "Constant":
                raise AssertionError(
                    f"Constant node '{node.name}' was not lifted to initializers"
                )
            if node.op_type == "Add":
                self.assertEqual(len(node.graph.initializers), 1)
                self.assertEqual(
                    node.graph.initializers["val_0"].const_value,
                    then_constant_tensor,
                )
            if node.op_type == "Mul":
                self.assertEqual(len(node.graph.initializers), 1)
                self.assertEqual(
                    node.graph.initializers["val_0"].const_value,
                    else_constant_tensor,
                )
