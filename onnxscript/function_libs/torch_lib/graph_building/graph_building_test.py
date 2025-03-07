# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Test cases for graph building functionality."""

# mypy: disable-error-code="arg-type,type-arg,valid-type"
from __future__ import annotations

import os
import sys
import unittest

import torch

import onnxscript
import onnxscript.testing
from onnxscript import FLOAT, evaluator
from onnxscript import opset18 as op
from onnxscript._internal import version_utils
from onnxscript.function_libs.torch_lib import graph_building, ops

IS_WINDOWS = os.name == "nt"


class TestTorchScriptTracingEvaluator(unittest.TestCase):
    def setUp(self):
        self.opset_version = 18
        # TODO: Add test for initializer. Currently skipped since to `assert_isomorphic`
        # does not check for initializers.
        self.onnxscript_graph = graph_building.TorchScriptGraph()
        self.tracer = graph_building.TorchScriptTracingEvaluator(self.onnxscript_graph)

    def test_torchscript_tensor_keeps_torch_device(self):
        x_tensor = torch.ones((1, 2, 3), dtype=torch.float32)
        x = self.onnxscript_graph.add_input(
            "x", x_tensor.shape, x_tensor.dtype, x_tensor.device
        )
        self.assertEqual(x.device, x_tensor.device)

        x.device = torch.device("cuda")
        self.assertEqual(x.device, torch.device("cuda"))

    def test_traced_constant_op_is_same_as_compiled_graph(self):
        """Test for op.Constant created in graph builder"""
        with evaluator.default_as(self.tracer):
            output = op.Constant(value_float=0.5)

        self.onnxscript_graph.register_outputs(output)
        traced = self.onnxscript_graph.to_model_proto(self.opset_version)

        @onnxscript.script()
        def expected_model():
            return op.Constant(value_float=0.5)

        expected = expected_model.to_model_proto()

        onnxscript.testing.assert_isomorphic(traced, expected)

    @unittest.expectedFailure  # Failed after #1836. Fix me.
    def test_traced_graph_on_single_node_is_same_as_compiled_graph(self):
        aten_elu = ops.nn.aten_elu

        x_tensor = torch.ones((1, 2, 3), dtype=torch.float32)
        x = self.onnxscript_graph.add_input("x", x_tensor.shape, x_tensor.dtype)
        with evaluator.default_as(self.tracer):
            output = aten_elu(x)

        self.onnxscript_graph.register_outputs(output)
        traced = self.onnxscript_graph.to_model_proto(self.opset_version)

        @onnxscript.script(default_opset=op)
        def expected_model(x: FLOAT[1, 2, 3]):
            return aten_elu(x)

        expected = expected_model.to_model_proto()

        onnxscript.testing.assert_isomorphic(traced, expected)

    @unittest.expectedFailure  # The scripted version does not have output type
    def test_traced_graph_on_single_node_multi_output_is_same_as_compiled_graph(self):
        aten_topk = ops.core.aten_topk

        x_tensor = torch.ones((1, 2, 3), dtype=torch.float32)
        x = self.onnxscript_graph.add_input("x", x_tensor.shape, x_tensor.dtype)
        with evaluator.default_as(self.tracer):
            output = aten_topk(x, 2)

        self.onnxscript_graph.register_outputs(output)
        traced = self.onnxscript_graph.to_model_proto(self.opset_version)

        @onnxscript.script(default_opset=op)
        def expected_model(x: FLOAT[1, 2, 3]):
            values, indices = aten_topk(x, 2)
            return values, indices

        expected = expected_model.to_model_proto()
        onnxscript.testing.assert_isomorphic(traced, expected)

    @unittest.expectedFailure  # abs is traced now
    def test_model_local_function_constructed_by_traced_graph_is_same_as_compiled_graph(
        self,
    ):
        aten_abs = ops.core.aten_abs
        aten_elu = ops.nn.aten_elu

        inner_graph = graph_building.TorchScriptGraph(domain_name="test_domain")
        inner_tracer = graph_building.TorchScriptTracingEvaluator(inner_graph)

        x_tensor = torch.ones((1, 2, 3), dtype=torch.float32)
        x = inner_graph.add_input("x", x_tensor.shape, x_tensor.dtype)
        with evaluator.default_as(inner_tracer):
            output = aten_abs(x)
        inner_graph.register_outputs(output)

        outer_graph = graph_building.TorchScriptGraph()
        outer_tracer = graph_building.TorchScriptTracingEvaluator(outer_graph)
        x_tensor = torch.ones((1, 2, 3), dtype=torch.float32)
        x = outer_graph.add_input("x", x_tensor.shape, x_tensor.dtype)
        with evaluator.default_as(outer_tracer):
            output = aten_elu(x)
        output = outer_graph.add_module_call("inner", inner_graph, (output,))
        outer_graph.register_outputs(output)
        traced = outer_graph.to_model_proto(self.opset_version)

        @onnxscript.script(
            opset=onnxscript.values.Opset("test_domain", 1),
            default_opset=op,
        )
        def inner(x: FLOAT[1, 2, 3]):
            return aten_abs(x)

        @onnxscript.script(default_opset=op)
        def outer(x: FLOAT[1, 2, 3]):
            output = aten_elu(x)
            return inner(output)

        expected = outer.to_model_proto()
        onnxscript.testing.assert_isomorphic(traced, expected)

    def test_add_input_with_optionaltype_does_not_raise_torch_internal_error(self):
        graph = graph_building.TorchScriptGraph()
        x = graph.add_input(input_name=None)
        with evaluator.default_as(self.tracer):
            _ = x.shape


class TestTorchScriptGraph(unittest.TestCase):
    def test_add_initializer_raises_when_the_same_name_used_for_different_tensors(self):
        graph = graph_building.TorchScriptGraph()
        graph.add_initializer("x", torch.ones((1, 2, 3), dtype=torch.float32))
        with self.assertRaises(ValueError):
            graph.add_initializer("x", torch.ones((1, 2, 3), dtype=torch.float32))

    def test_add_initializer_allows_adding_the_same_tensor_twice_using_same_name(self):
        graph = graph_building.TorchScriptGraph()
        x_tensor = torch.ones((1, 2, 3), dtype=torch.float32)
        graph.add_initializer("x", x_tensor)
        graph.add_initializer("x", x_tensor)


class _MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


@unittest.skipIf(
    IS_WINDOWS and version_utils.torch_older_than("2.3"),
    "dynamo_export not supported on Windows in PyTorch<2.3",
)
@unittest.skipIf(
    sys.version_info > (3, 11),
    "dynamo_export not supported due to torch.compile not functional for python>3.11",
)
class TestModelSaving(unittest.TestCase):
    def test_save_initializer_to_files_for_large_model(self):
        # # of model parameters:
        #  input_size x hidden_size + hidden_size +
        #  hidden_size x output_size + output_size
        #  ~= 3GB below
        batch_size, input_size, hidden_size, output_size = 1, 4, 50000000, 10
        model = _MLP(input_size, hidden_size, output_size)
        x = torch.randn(batch_size, input_size)

        model_proto = torch.onnx.dynamo_export(model, x).model_proto
        # Assert model is larger than 2GB (~=3GB)
        self.assertGreater(model_proto.ByteSize(), 2**31)

    def test_input_output_and_initializer_are_not_stored_in_value_info(self):
        batch_size, input_size, hidden_size, output_size = 1, 4, 5, 10
        model = _MLP(input_size, hidden_size, output_size)
        x = torch.randn(batch_size, input_size)

        model_proto = torch.onnx.dynamo_export(model, x).model_proto
        v_names = {v.name for v in model_proto.graph.value_info}

        for i in model_proto.graph.input:
            self.assertNotIn(i.name, v_names)
        for o in model_proto.graph.output:
            self.assertNotIn(o.name, v_names)
        for i in model_proto.graph.initializer:
            self.assertNotIn(i.name, v_names)

    @unittest.skipIf(
        not version_utils.torch_older_than("2.4"),
        "PyTorch 2.4-preview optimizes the functions away",
    )
    def test_experimental_function_value_info_are_stored_in_graph_value_info(self):
        batch_size, input_size, hidden_size, output_size = 1, 4, 5, 10
        model = _MLP(input_size, hidden_size, output_size)
        x = torch.randn(batch_size, input_size)

        model_proto = torch.onnx.dynamo_export(model, x).model_proto
        v_names = {v.name for v in model_proto.graph.value_info}
        torch_functions = [
            f for f in model_proto.functions if f.domain.startswith("pkg.torch")
        ]
        self.assertNotEqual(len(torch_functions), 0)
        for f in torch_functions:
            for n in f.node:
                for i in n.input:
                    self.assertIn(f"{f.domain}::{f.name}/{i}", v_names)
                for o in n.output:
                    self.assertIn(f"{f.domain}::{f.name}/{o}", v_names)


if __name__ == "__main__":
    unittest.main()
