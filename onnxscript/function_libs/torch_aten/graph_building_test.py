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
        aten_amax = ops.core.aten_amax

        x = self.onnxscript_graph.add_input("x", torch.randn(2, 3, 4))
        with evaluator.default_as(self.tracer):
            output = aten_amax(x, dim=0, keepdim=False)

        self.onnxscript_graph.register_outputs(output)

        @onnxscript.script()
        def expected_model(x: FLOAT[2, 3, 4]):
            return aten_amax(x, dim=0, keepdim=False)

        traced = self.to_model_proto()
        expected = expected_model.to_model_proto()
        import onnx

        expected = onnx.shape_inference.infer_shapes(expected)
        onnx.checker.check_model(expected, full_check=True)
        onnx.checker.check_model(traced, full_check=True)
        print("traced: ", traced)
        print("expected model: ", expected)
        self.assertSame(traced, expected)

    def test_inputs_created_as_constant_op(self):
        """Test for _add_constant_to_graph function"""
        pass
