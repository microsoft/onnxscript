"""Test cases for graph building functionality."""
from __future__ import annotations

import collections
import typing
import warnings
from typing import Any, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.checker
import onnx.defs
import onnx.shape_inference
import torch
from beartype import beartype
from torch.onnx import _type_utils
import expecttest
import onnxscript
from onnxscript import evaluator, script
from onnxscript.function_libs.torch_aten import graph_building, ops
from onnxscript import tensor as onnxscript_tensor

class TestCase(expecttest.TestCase):

    # TODO(titaiwang): opset version
        
    def test_attribute_split_logics(self):
        """Test for _split_args_kwargs_to_input_attr function"""

        # initialize graph
        onnxscript_graph = graph_building.TorchScriptGraph()
        tracer = graph_building.TorchScriptTracingEvaluator(onnxscript_graph)
        # TODO(titaiwang): prepare input (fake tensor?)
        model_inputs = torch.randn(2, 3, 4)
        onnxscript_graph.add_input("model_input", model_inputs)
        # Ops
        with evaluator.default_as(tracer):
            # TODO(titaiwang): beartype
            model_outputs = ops.core.aten_add(model_inputs, model_inputs, 1, 2)
        # register output
        onnxscript_graph.register_outputs(model_outputs)
        # check
        onnx_model = onnxscript_graph.to_model_proto()
        onnx.check.check_model(onnx_model, full_check=True)

    def test_inputs_created_as_constant_op(self):
        """Test for _add_constant_to_graph function"""
        pass
        
    def test_constant_op_with_evaluator(self):
        """Test for op.Constant created in graph builder"""
        from onnxscript import opset18 as op
        # initialize graph
        onnxscript_graph = graph_building.TorchScriptGraph()
        tracer = graph_building.TorchScriptTracingEvaluator(onnxscript_graph)
        with evaluator.default_as(tracer):
            # TODO(titaiwang): _eval/eval
            op.Constant(value_float=0.5)
        onnx_model = onnxscript_graph.to_model_proto()
        onnx.checker.check_model(onnx_model, full_check=True)