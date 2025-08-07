# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import numpy as np

from onnxscript import ir
from onnxscript.ir.passes.common.graph_extration import ExtractGraphPass


class TestExtractGraphPass(unittest.TestCase):
    def test_extract_subgraph(self):
        inputs = [
            ir.Value(name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))),
            ir.Value(name="input_b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))),
        ]

        add_node = ir.node("Add", inputs=inputs)
        mul_node = ir.node("Mul", inputs=[add_node.outputs[0], inputs[1]])

        model = ir.Model(
            graph=ir.Graph(
                inputs=inputs,
                outputs=mul_node.outputs,
                nodes=[add_node, mul_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Perform extract graph pass
        extract_pass = ExtractGraphPass(input_names=["input_a"], output_names=[mul_node.outputs[0].name])
        result = extract_pass(model)
        self.assertTrue(result.modified)
        self.assertEqual(len(result.model.graph.nodes), 2)
        self.assertEqual(result.model.graph.nodes[0].op_type, "Add")
        self.assertEqual(result.model.graph.nodes[1].op_type, "Mul")

    def test_extract_subgraph_with_initializers(self):
        inputs = [
            ir.Value(name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))),
            ir.Value(name="input_b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))),
        ]

        constant_tensor = ir.tensor(np.random.rand(2, 3).astype(ir.DataType.FLOAT.numpy()))
        const_node = ir.node(
            "Constant", inputs=[], attributes={"value": constant_tensor}, num_outputs=1
        )
        add_node = ir.node("Add", inputs=[inputs[0], const_node.outputs[0]])
        mul_node = ir.node("Mul", inputs=[add_node.outputs[0], inputs[1]])

        model = ir.Model(
            graph=ir.Graph(
                inputs=inputs,
                outputs=mul_node.outputs,
                nodes=[const_node, add_node, mul_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Perform extract graph pass
        extract_pass = ExtractGraphPass(input_names=["input_a"], output_names=[mul_node.outputs[0].name])
        result = extract_pass(model)
        self.assertTrue(result.modified)
        self.assertEqual(len(result.model.graph.nodes), 3)
        self.assertEqual(result.model.graph.nodes[0].op_type, "Constant")
        self.assertEqual(result.model.graph.nodes[1].op_type, "Add")
        self.assertEqual(result.model.graph.nodes[2].op_type, "Mul")

    def test_extract_subgraph_with_subgraph(self):
        input_value = ir.Value(
            name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))
        )

        then_constant_tensor = ir.tensor(np.random.rand(2, 3).astype(np.float32))
        then_const_node = ir.node(
            "Constant", inputs=[], attributes={"value": then_constant_tensor}, num_outputs=1
        )
        add_node = ir.node("Add", inputs=[input_value, then_const_node.outputs[0]])
        then_graph = ir.Graph(
            inputs=[input_value],
            outputs=[add_node.outputs[0]],
            nodes=[then_const_node, add_node],
            opset_imports={"": 20},
        )
        else_constant_tensor = ir.tensor(np.random.rand(2, 3).astype(np.float32))
        else_const_node = ir.node(
            "Constant", inputs=[], attributes={"value": else_constant_tensor}, num_outputs=1
        )
        mul_node = ir.node("Mul", inputs=[input_value, else_const_node.outputs[0]])
        else_graph = ir.Graph(
            inputs=[input_value],
            outputs=[mul_node.outputs[0]],
            nodes=[else_const_node, mul_node],
            opset_imports={"": 20},
        )
        cond_node = ir.node(
            "If",
            inputs=[input_value],
            attributes={"then_branch": then_graph, "else_branch": else_graph},
            num_outputs=1,
        )
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

        # Perform extract graph pass
        extract_pass = ExtractGraphPass(input_names=["input"], output_names=[cond_node.outputs[0].name])
        result = extract_pass(model)
        self.assertTrue(result.modified)
        self.assertEqual(len(result.model.graph.nodes), 1)
        self.assertEqual(result.model.graph.nodes[0].op_type, "If")
        self.assertEqual(len(result.model.graph.nodes[0].attributes["then_branch"].nodes), 2)
        self.assertEqual(len(result.model.graph.nodes[0].attributes["else_branch"].nodes), 2)

    def test_extract_partial_subgraph(self):
        inputs = [
            ir.Value(name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))),
            ir.Value(name="input_b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))),
        ]

        add_node = ir.node("Add", inputs=inputs)
        mul_node = ir.node("Mul", inputs=[add_node.outputs[0], inputs[1]])
        sub_node = ir.node("Sub", inputs=[mul_node.outputs[0], inputs[0]])

        model = ir.Model(
            graph=ir.Graph(
                inputs=inputs,
                outputs=sub_node.outputs,
                nodes=[add_node, mul_node, sub_node],
                opset_imports={"": 20},
            ),
            ir_version=10,
        )

        # Perform extract graph pass
        extract_pass = ExtractGraphPass(input_names=["input_a"], output_names=[mul_node.outputs[0].name])
        result = extract_pass(model)
        self.assertTrue(result.modified)
        self.assertEqual(len(result.model.graph.nodes), 2)
        self.assertEqual(result.model.graph.nodes[0].op_type, "Add")
        self.assertEqual(result.model.graph.nodes[1].op_type, "Mul")


if __name__ == "__main__":
    unittest.main()
