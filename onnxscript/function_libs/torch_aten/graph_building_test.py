"""Test cases for graph building functionality."""
# mypy: disable-error-code="arg-type,type-arg,valid-type"
from __future__ import annotations

import unittest

import torch

import onnxscript
import onnxscript.testing
from onnxscript import FLOAT, evaluator
from onnxscript import opset17 as op
from onnxscript.function_libs.torch_aten import graph_building, ops
from onnxscript.tests.common import version_utils


@unittest.skipIf(version_utils.torch_older_than("2.0"), "torchscript in 1.13 not supported")
class TestTorchScriptTracingEvaluator(unittest.TestCase):
    def setUp(self):
        self.opset_version = 17
        self.onnxscript_graph = graph_building.TorchScriptGraph()
        self.tracer = graph_building.TorchScriptTracingEvaluator(self.onnxscript_graph)

    def to_model_proto(self):
        # TODO(titaiwang): initializer API
        return self.onnxscript_graph.to_model_proto(
            initializers={}, opset_version=self.opset_version
        )

    def test_traced_constant_op_is_same_as_compiled_graph(self):
        """Test for op.Constant created in graph builder"""
        with evaluator.default_as(self.tracer):
            output = op.Constant(value_float=0.5)

        self.onnxscript_graph.register_outputs(output)
        traced = self.to_model_proto()

        @onnxscript.script()
        def expected_model():
            return op.Constant(value_float=0.5)

        expected = expected_model.to_model_proto()

        onnxscript.testing.assert_isomorphic(traced, expected)

    def test_traced_graph_on_single_node_is_same_as_compiled_graph(self):
        aten_relu = ops.nn.aten_relu

        x = self.onnxscript_graph.add_input("x", torch.ones((1, 2, 3), dtype=torch.float32))
        with evaluator.default_as(self.tracer):
            output = aten_relu(x)

        self.onnxscript_graph.register_outputs(output)
        traced = self.to_model_proto()

        @onnxscript.script(default_opset=op)
        def expected_model(x: FLOAT[1, 2, 3]):
            return aten_relu(x)

        expected = expected_model.to_model_proto()

        onnxscript.testing.assert_isomorphic(traced, expected)

    @unittest.expectedFailure  # The scripted version does not have output type
    def test_traced_graph_on_single_node_multi_output_is_same_as_compiled_graph(self):
        aten_topk = ops.core.aten_topk

        x = self.onnxscript_graph.add_input("x", torch.ones((1, 2, 3), dtype=torch.float32))
        with evaluator.default_as(self.tracer):
            output = aten_topk(x, 2)

        self.onnxscript_graph.register_outputs(output)
        traced = self.to_model_proto()

        @onnxscript.script(default_opset=op)
        def expected_model(x: FLOAT[1, 2, 3]):
            values, indices = aten_topk(x, 2)
            return values, indices

        expected = expected_model.to_model_proto()
        onnxscript.testing.assert_isomorphic(traced, expected)


if __name__ == "__main__":
    unittest.main()
