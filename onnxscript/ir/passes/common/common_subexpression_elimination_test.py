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
    def check_graph(self, model: ir.Model, inputs: list[ir.Value], delta_nodes: list[int]):
        """Check if the model applied the CSE pass correctly.

        Args:
            model (ir.Model): The model to check.
            inputs (list[ir.Value]): The inputs to the model.
            delta_nodes (list[int]): The expected change in the number of nodes in the model.
                                     The length of this list should match the number of graphs
                                     in the model. (to support subgraphs in the future)

        Raises:
            AssertionError: If the model does not match the expected number of nodes or outputs.

        """
        assert len(list(model.graphs())) == len(delta_nodes)
        # Log all results from the original model.
        # 1. model graph node counts
        original_graphs_node_count = np.array([graph.num_nodes() for graph in model.graphs()])
        model_proto = ir.serde.serialize_model(model)

        # 2. model outputs
        ort_inputs = {
            k.name: np.random.rand(*v.shape).astype(np.float32)
            for k, v in zip(model.graph.inputs, inputs)
        }
        original_model_session = ort.InferenceSession(model_proto.SerializeToString())
        original_model_results = original_model_session.run(None, ort_inputs)

        result = common_subexpression_elimination.CommonSubexpressionEliminationPass()(model)

        result_graphs_node_count = np.array([graph.num_nodes() for graph in model.graphs()])
        # Check if the number of nodes in the model is correct
        self.assertTrue(
            np.array_equal(
                original_graphs_node_count, np.add(result_graphs_node_count, delta_nodes)
            )
        )
        self.assertEqual(
            result.modified, any(original_graphs_node_count > result_graphs_node_count)
        )

        result_proto = ir.serde.serialize_model(result.model)
        result_session = ort.InferenceSession(result_proto.SerializeToString())
        result_results = result_session.run(None, ort_inputs)

        # Check if the models produce the same output
        # with the same inputs
        for idx, original_model_result in enumerate(original_model_results):
            np.testing.assert_allclose(
                original_model_result, result_results[idx], rtol=1e-5, atol=1e-5
            )

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

        self.check_graph(model, [np.random.rand(2, 2)], delta_nodes=[2])

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
        self.check_graph(model, [np.random.rand(1)], delta_nodes=[3])

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
        self.check_graph(model, [np.random.rand(2, 2)], delta_nodes=[3])

    def test_the_ops_with_the_same_inputs_but_different_attributes_are_not_csed(self):
        """Test if the ops with the same inputs but different attributes are not CSEd.

        def f(x):
            a = x.sum()
            b = x.sum(keepdims=True)
            c = x.sum()
            d = x.sum(keepdims=True)
            return a + b + c + d

        x = torch.randn(2, 2)

        """

        @script()
        def test_model(x: FLOAT[2, 2]) -> FLOAT[2, 2]:
            a = op.ReduceSum(x, keepdims=False)
            b = op.ReduceSum(x, keepdims=True)
            return a + b

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        self.check_graph(model, [np.random.rand(2, 2)], delta_nodes=[0])

    def test_control_flow_if_ops_are_not_csed_as_graph_attr_is_not_matched(self):
        """Test if control flow ops are not CSEd.

        def f(a, b):
            rank = a.rank()
            if rank == 2:
                result1 = a - b
            else:
                result1 = a + b
            if rank == 2:
                result2 = a - b
            else:
                result2 = a + b
            return result1 + result2

        x = torch.randn(2, 2)

        """

        @script()
        def test_model(a: FLOAT[2, 2], b: FLOAT[2, 2]) -> FLOAT[2, 2]:
            rank = op.Size(op.Shape(a))
            if rank == 2:
                result1 = a - b
            else:
                result1 = a + b
            if rank == 2:
                result2 = a - b
            else:
                result2 = a + b
            return result1 + result2

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        self.check_graph(
            model, [np.random.rand(2, 2), np.random.rand(2, 2)], delta_nodes=[0, 0, 0, 0, 0]
        )

    def test_the_nodes_following_control_flow_ops_are_csed(self):
        """Test if the nodes following control flow ops are CSEd.

        def f(a, b):
            rank = a.rank()
            if rank == 2:
                x = a - b
            else:
                x = a + b
            a = x.cos().sin()
            b = x.cos().sin()
            c = a + a
            d = b + b
            return c + d

            x = torch.randn(2, 2)

        """

        @script()
        def test_model(a: FLOAT[2, 2], b: FLOAT[2, 2]) -> FLOAT[2, 2]:
            rank = op.Size(op.Shape(a))
            if rank == 2:
                x = a - b
            else:
                x = a + b
            a = op.Sin(op.Cos(x))
            b = op.Sin(op.Cos(x))
            c = a + a
            d = b + b
            return c + d

        model_proto = test_model.to_model_proto()
        model = ir.serde.deserialize_model(model_proto)
        self.check_graph(
            model, [np.random.rand(2, 2), np.random.rand(2, 2)], delta_nodes=[3, 0, 0]
        )
