# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest
import numpy as np
from onnxscript import ir
from onnxscript.ir.passes.common import clear_metadata_and_docstring

class TestClearMetadataAndDocStringPass(unittest.TestCase):
    def test_pass_with_clear_metadata_and_docstring(self):
        # Create a model (node, graph, function) with metadata and docstring
        inputs = [
            ir.Value(name="input_a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))),
            ir.Value(name="input_b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 3))),
        ]
        add_node = ir.node("Add", inputs=inputs, num_outputs=1, metadata_props={"add_key": "add_value"}, doc_string="This is an Add node")
        mul_node = ir.node("Mul", inputs=[add_node.outputs[0], inputs[1]], num_outputs=1, metadata_props={"mul_key": "mul_value"}, doc_string="This is a Mul node")
        function = ir.Function(
            graph=ir.Graph(
                name="my_function",
                inputs=inputs,
                outputs=mul_node.outputs,
                nodes=[add_node, mul_node],
                opset_imports={"": 20},
                doc_string="This is a function docstring",
                metadata_props={"function_key": "function_value"},
            ),
            name="my_function",
            domain="my_domain",
            attributes=[]
        )
        # Create a model with the graph and function
        constant_tensor = ir.tensor(np.random.rand(2, 3).astype(ir.DataType.FLOAT.numpy()))
        const_node = ir.node(
            "Constant", inputs=[], attributes={"value": constant_tensor}, num_outputs=1, metadata_props={"const_key": "const_value"}, doc_string="This is a Constant node"
        )
        sub_node = ir.node("Sub", inputs=[function.outputs[0], const_node.outputs[0]], num_outputs=1, metadata_props={"sub_key": "sub_value"}, doc_string="This is a Sub node")
        model = ir.Model(
            graph=ir.Graph(
                inputs=inputs,
                outputs=sub_node.outputs,
                nodes=[const_node, sub_node],
                opset_imports={"": 20},
                doc_string="This is a graph docstring",
                metadata_props={"graph_key": "graph_value"},
            ),
            ir_version=10,
            functions=[function],

        )
        # Create a pass to clear metadata and docstring
        clear_pass = clear_metadata_and_docstring.ClearMetadataAndDocStringPass()
        # Apply the pass
        result = clear_pass(model)
        # Check that the pass was applied
        self.assertTrue(result.modified)
        # Check that the metadata and docstring were cleared
        self.assertEqual(model.graph.doc_string, None)
        self.assertEqual(model.graph.metadata_props, {})
        for node in model.graph:
            self.assertEqual(node.metadata_props, {})
            self.assertEqual(node.doc_string, None)
        # Check that the function docstring and metadata were cleared
        self.assertEqual(function.doc_string, None)
        self.assertEqual(function.metadata_props, {})