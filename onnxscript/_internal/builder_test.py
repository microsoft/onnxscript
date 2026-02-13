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


if __name__ == "__main__":
    unittest.main()
