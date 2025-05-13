# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnxruntime as ort

from onnxscript import ir
from onnxscript.ir.passes.common import common_subexpression_elimination


class TestCommonSubexpressionEliminationPass(unittest.TestCase):
    def check_graph(self, model: ir.Model, inputs: list[ir.Value], delta_nodes: int = 0):
        """Check if the model applied the CSE pass correctly."""
        result = common_subexpression_elimination.CommonSubexpressionEliminationPass()(model)
        import pdb

        pdb.set_trace()
        # Check if the number of nodes in the model is correct
        self.assertEqual(len(model.graph), len(result.model.graph) + delta_nodes)
        self.assertEqual(result.modified, len(model.graph) > len(result.model))

        model_proto = ir.serde.serialize_model(model)
        result_proto = ir.serde.serialize_model(result.model)
        # Check if the models produce the same output
        # with the same inputs
        ort_inputs = {
            k.name: np.random.rand(*v.shape) for k, v in zip(model.graph.inputs, inputs)
        }
        ort_session = ort.InferenceSession(model_proto.SerializeToString())
        ort_result = ort_session.run(None, ort_inputs)
        result_session = ort.InferenceSession(result_proto.SerializeToString())
        result_result = result_session.run(None, ort_inputs)
        for i in range(len(ort_result)):
            np.testing.assert_allclose(ort_result[i], result_result[i], rtol=1e-5, atol=1e-5)

    def test_two_branches_with_the_same_operations_is_csed(self):
        """Test if two branches with the same operations are CSEd.

        def test_simple(self):
            def f(x):
                a = x.cos()
                b = x.cos()
                c = a + a
                d = b + b
                return c + d

            x = torch.randn(2, 2)
        """
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2, 2)))
        a = ir.node("Cos", inputs=[x], num_outputs=1)
        b = ir.node("Cos", inputs=[x], num_outputs=1)
        c = ir.node("Add", inputs=[a.outputs[0], a.outputs[0]], num_outputs=1)
        d = ir.node("Add", inputs=[b.outputs[0], b.outputs[0]], num_outputs=1)
        e = ir.node("Add", inputs=[c.outputs[0], d.outputs[0]], num_outputs=1)
        model = ir.Model(
            graph=ir.Graph(
                name="test_graph",
                inputs=[x],
                outputs=[e.outputs[0]],
                nodes=[a, b, c, d, e],
            ),
            ir_version=10,
        )
        self.check_graph(model, [np.random.rand(2, 2)], delta_nodes=2)
