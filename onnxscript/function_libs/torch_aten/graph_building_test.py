"""Test cases for graph building functionality."""
from __future__ import annotations

import onnx
import onnx.checker
import onnx.defs
import onnx.shape_inference
import torch

import onnxscript
from onnxscript import evaluator, FLOAT
from onnxscript.function_libs.torch_aten import graph_building, ops
from onnxscript.tests.common import testutils


class TestTorchScriptTracingEvaluator(testutils.TestBase):
    def setUp(self):
        self.opset_version = 17
        self.onnxscript_graph = graph_building.TorchScriptGraph()
        self.tracer = graph_building.TorchScriptTracingEvaluator(self.onnxscript_graph)

    # TODO(titaiwang): opset version
    def test_attribute_split_logics(self):
        """Test for _split_args_kwargs_to_input_attr function"""
        pass

    def test_traced_graph_on_single_node_is_isomorphic_with_compiled_graph(self):
        # initialize graph
        input_ = torch.randn(2, 3, 4)
        x = self.onnxscript_graph.add_input("x", input_)
        with evaluator.default_as(self.tracer):
            output = ops.core.aten_amax(x, dim=0, keepdim=0)
        self.onnxscript_graph.register_outputs(output)
        # self.onnxscript_graph.register_outputs(model_outputs2)
        # self.onnxscript_graph.register_outputs(model_outputs3)
        aten_amax = ops.core.aten_amax
        @onnxscript.script()
        def amax_model(x: FLOAT[2, 3, 4]):
            return aten_amax(x, dim=0, keepdim=0)
        # check model
        onnx_model = self.onnxscript_graph.to_model_proto(
            # TODO(titaiwang): initializer API
            initializers={},
            opset_version=self.opset_version
        )
        #onnx.check.check_model(onnx_model, full_check=True)
        expected_model = amax_model.to_model_proto()
        print("\nonnx_model: ", onnx_model)
        print("\nexpected model: ", expected_model)
        self.assertSame(onnx_model, expected_model)

    def test_inputs_created_as_constant_op(self):
        """Test for _add_constant_to_graph function"""
        pass

    def test_constant_op_with_evaluator(self):
        """Test for op.Constant created in graph builder"""
        from onnxscript import opset17 as op
        with evaluator.default_as(self.tracer):
            output = op.Constant(value_float=0.5)
        self.onnxscript_graph.register_outputs(output)
        onnx_model = self.onnxscript_graph.to_model_proto(initializers={}, opset_version=self.opset_version)
        # onnx.checker.check_model(onnx_model, full_check=True)
        @onnxscript.script()
        def constant_model():
            return op.Constant(value_float=0.5)
        
        expected_model = constant_model.to_model_proto()
        print("\nonnx_model: ", onnx_model)
        print("\nexpected model: ", expected_model)
        self.assertSame(onnx_model, expected_model)
