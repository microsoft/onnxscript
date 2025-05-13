# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnxruntime as ort

from onnxscript import FLOAT, ir, script
from onnxscript import opset18 as op
from onnxscript.ir.passes.common import common_subexpression_elimination


class TestCommonSubexpressionEliminationPass(unittest.TestCase):
    def check_graph(self, model: ir.Model, inputs: list[ir.Value], delta_nodes: int = 0):
        """Check if the model applied the CSE pass correctly."""
        result = common_subexpression_elimination.CommonSubexpressionEliminationPass()(model)
        # Check if the number of nodes in the model is correct
        self.assertEqual(len(model.graph), len(result.model.graph) + delta_nodes)
        self.assertEqual(result.modified, len(model.graph) > len(result.model.graph))

        model_proto = ir.serde.serialize_model(model)
        result_proto = ir.serde.serialize_model(result.model)
        # Check if the models produce the same output
        # with the same inputs
        ort_inputs = {
            k.name: np.random.rand(*v.shape).astype(np.float32)
            for k, v in zip(model.graph.inputs, inputs)
        }
        ort_session = ort.InferenceSession(model_proto.SerializeToString())
        ort_results = ort_session.run(None, ort_inputs)
        result_session = ort.InferenceSession(result_proto.SerializeToString())
        result_results = result_session.run(None, ort_inputs)
        for idx, ort_result in enumerate(ort_results):
            np.testing.assert_allclose(ort_result, result_results[idx], rtol=1e-5, atol=1e-5)

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

        @script()
        def test_model(x: FLOAT[2, 2]) -> FLOAT[2, 2]:
            a = op.Cos(x)
            b = op.Cos(x)
            c = a + a
            d = b + b
            return c + d

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)

        self.check_graph(model, [np.random.rand(2, 2)], delta_nodes=2)

    def test_more_operations_in_two_branches_with_the_same_operations_is_csed(self):
        """Test if two branches with the same operations are CSEd.

        def test_simple(self):
            def f(x):
                a = x.cos().sin()
                b = x.cos().sin()
                c = a + a
                d = b + b
                return c + d

            x = torch.randn(2, 2)
        """

        @script()
        def test_model(x: FLOAT[1]) -> FLOAT[1]:
            a = op.Sin(op.Cos(x))
            b = op.Sin(op.Cos(x))
            c = a + a
            d = b + b
            return c + d

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        self.check_graph(model, [np.random.rand(1)], delta_nodes=3)

    def test_multiple_same_ops_with_attributes_are_csed(self):
        """Test if multiple same ops are CSEd.

        def f(x):
            a = x.sum()
            b = x.sum()
            c = x.sum()
            d = x.sum()
            return a + b + c + d

        x = torch.randn(2, 2)

        """

        @script()
        def test_model(x: FLOAT[2, 2]) -> FLOAT[2, 2]:
            a = op.ReduceSum(x, keepdims=False)
            b = op.ReduceSum(x, keepdims=False)
            c = op.ReduceSum(x, keepdims=False)
            d = op.ReduceSum(x, keepdims=False)
            return a + b + c + d

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        self.check_graph(model, [np.random.rand(2, 2)], delta_nodes=3)
