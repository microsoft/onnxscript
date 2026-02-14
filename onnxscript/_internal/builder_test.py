# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import unittest
from typing import Sequence

import onnx_ir as ir
import onnxscript._internal.builder as builder


def _build(trace_function, input_types: Sequence[ir.TypeAndShape], output_types: Sequence[ir.TypeAndShape]) -> ir.Model:

    graph = ir.Graph(
        name="test_model",
        inputs=[],
        outputs=[],
        nodes=[],
        opset_imports={"": 23},
    )

    onnx_model = ir.Model(graph=graph, ir_version=10)

    for i, input_type in enumerate(input_types):
        input_name = f"input_{i}"
        graph.inputs.append(ir.Value(name=input_name, type=input_type))

    graph_builder = builder.GraphBuilder(graph, is_function=False)
    outputs = trace_function(graph_builder.op, *graph.inputs)
    if not isinstance(outputs, Sequence):
        outputs = [outputs]
    if len(outputs) != len(output_types):
        raise ValueError(f"Expected {len(output_types)} outputs, but got {len(outputs)}.")
    for output, output_type in zip(outputs, output_types):
        output.type = output_type.type  # TODO: need merge_type method in ir.Value
        output.merge_shapes(output_type.shape)

    graph.outputs.extend(outputs)

    return onnx_model


class GraphBuilderTest(unittest.TestCase):
    def test_builder_basic(self):
        def _add_mul_add(op: builder.OpBuilder, x: ir.Value, y: ir.Value) -> ir.Value:
            t1 = op.Add(x, y)
            t2 = op.Mul(x, y)
            z = op.Add(t1, t2)
            return z

        float_2d = ir.TypeAndShape(ir.TensorType(ir.DataType.FLOAT), ir.Shape([3, 4]))
        model = _build(
            _add_mul_add,
            input_types=[float_2d, float_2d],
            output_types=[float_2d],
        )
        graph = model.graph
        # Expect exactly 3 nodes: Add, Mul, Add
        op_types = [node.op_type for node in graph]
        self.assertEqual(op_types, ["Add", "Mul", "Add"])

        # Verify inputs and outputs
        self.assertEqual(len(graph.inputs), 2)
        self.assertEqual(len(graph.outputs), 1)

        # Verify the connectivity: final Add takes outputs of the first Add and Mul
        nodes = list(graph)
        add1, mul, add2 = nodes
        self.assertEqual(list(add2.inputs), [add1.outputs[0], mul.outputs[0]])

    def test_value_naming(self):
        """Test that output names can be specified via the _outputs option."""
        def _add_with_custom_names(op: builder.OpBuilder, x: ir.Value, y: ir.Value) -> ir.Value:
            # Specify custom names for output values
            t1 = op.Add(x, y, _outputs=["add_result"])
            t2 = op.Mul(x, y, _outputs=["mul_result"])
            z = op.Add(t1, t2, _outputs=["final_add"])
            return z

        float_2d = ir.TypeAndShape(ir.TensorType(ir.DataType.FLOAT), ir.Shape([3, 4]))
        model = _build(
            _add_with_custom_names,
            input_types=[float_2d, float_2d],
            output_types=[float_2d],
        )
        graph = model.graph
        
        # Verify that the nodes have outputs with the specified names
        nodes = list(graph)
        self.assertEqual(len(nodes), 3)
        
        # Check output names
        self.assertEqual(nodes[0].outputs[0].name, "add_result")
        self.assertEqual(nodes[1].outputs[0].name, "mul_result")
        self.assertEqual(nodes[2].outputs[0].name, "final_add")
        
        # Verify the final output has the correct name
        self.assertEqual(len(graph.outputs), 1)
        self.assertEqual(graph.outputs[0].name, "final_add")

    def test_output_naming_strategy(self):
        """Test the default naming strategy for generated output values using op_type_output format."""
        def _ops_with_default_names(op: builder.OpBuilder, x: ir.Value, y: ir.Value) -> ir.Value:
            # Single output operations should be named {op_type}_output
            t1 = op.Add(x, y)
            t2 = op.Mul(x, y)
            z = op.Add(t1, t2)
            return z

        float_2d = ir.TypeAndShape(ir.TensorType(ir.DataType.FLOAT), ir.Shape([3, 4]))
        model = _build(
            _ops_with_default_names,
            input_types=[float_2d, float_2d],
            output_types=[float_2d],
        )
        graph = model.graph
        
        # Verify the nodes use the new naming strategy
        nodes = list(graph)
        self.assertEqual(len(nodes), 3)
        
        # Check output names follow the {op_type}_output pattern for single outputs
        self.assertEqual(nodes[0].outputs[0].name, "Add_output")
        self.assertEqual(nodes[1].outputs[0].name, "Mul_output")
        self.assertEqual(nodes[2].outputs[0].name, "Add_output")
        
        # Verify the final output has the correct name
        self.assertEqual(len(graph.outputs), 1)
        self.assertEqual(graph.outputs[0].name, "Add_output")

    def test_node_naming_strategy(self):
        """Test the node naming strategy using op_type_node_count format with hierarchical naming."""
        float_2d = ir.TypeAndShape(ir.TensorType(ir.DataType.FLOAT), ir.Shape([3, 4]))

        graph = ir.Graph(
            name="test_model",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 23},
        )

        for i in range(2):
            input_name = f"input_{i}"
            graph.inputs.append(ir.Value(name=input_name, type=float_2d))

        graph_builder = builder.GraphBuilder(graph, is_function=False)
        x, y = graph.inputs

        # Test node and value naming at root level
        t1 = graph_builder.op.Add(x, y)
        self.assertEqual(t1.name, "Add_output")
        self.assertEqual(t1.producer().name, "Add_node_0")

        t2 = graph_builder.op.Mul(t1, y)
        self.assertEqual(t2.name, "Mul_output")
        self.assertEqual(t2.producer().name, "Mul_node_1")

        # Test node and value naming with hierarchical context prefix
        graph_builder.push_module("layer1")
        t3 = graph_builder.op.Add(t2, x)
        self.assertEqual(t3.name, "layer1.Add_output")
        self.assertEqual(t3.producer().name, "layer1.Add_node_2")

        # Test nested hierarchical context
        graph_builder.push_module("attention")
        t4 = graph_builder.op.Mul(t3, y)
        self.assertEqual(t4.name, "layer1.attention.Mul_output")
        self.assertEqual(t4.producer().name, "layer1.attention.Mul_node_3")

        # Pop back to layer1 and verify naming continues correctly
        graph_builder.pop_module()
        t5 = graph_builder.op.Add(t4, x)
        self.assertEqual(t5.name, "layer1.Add_output")
        self.assertEqual(t5.producer().name, "layer1.Add_node_4")

        # Pop back to root context
        graph_builder.pop_module()
        t6 = graph_builder.op.Mul(t5, y)
        self.assertEqual(t6.name, "Mul_output")
        self.assertEqual(t6.producer().name, "Mul_node_5")


if __name__ == "__main__":
    unittest.main()
