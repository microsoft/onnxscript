"""Test cases for graph building functionality."""
from __future__ import annotations

import onnx.checker
import onnx.defs
import onnx.shape_inference
import torch

import onnxscript
from onnxscript import FLOAT, evaluator
from onnxscript import opset17 as op
from onnxscript.function_libs.torch_aten import graph_building, ops
from onnxscript.tests.common import testutils


class TestTorchScriptTracingEvaluator(testutils.TestBase):
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

        self.assertSame(traced, expected)

    def test_traced_graph_on_single_node_is_same_as_compiled_graph(self):
        aten_gelu = ops.nn.aten_gelu

        x = self.onnxscript_graph.add_input("x", torch.ones((1, 2, 3), dtype=torch.float32))
        with evaluator.default_as(self.tracer):
            output = aten_gelu(x, approximate="tanh")

        self.onnxscript_graph.register_outputs(output)
        traced = self.to_model_proto()

        @onnxscript.script()
        def expected_model(x: FLOAT[1, 2, 3]):
            return aten_gelu(x, approximate="tanh")

        expected = expected_model.to_model_proto()

        self.assertSame(traced, expected)
