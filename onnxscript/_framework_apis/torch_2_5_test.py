# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx
import onnx_ir as ir

from onnxscript._framework_apis import torch_2_5


def _make_model(identity_input: str) -> ir.Model:
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", [identity_input], ["y"])],
        "graph",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )
    return ir.serde.deserialize_model(model)


class CheckModelTest(unittest.TestCase):
    def test_check_model_accepts_valid_model(self):
        torch_2_5.check_model(_make_model("x"))

    def test_check_model_rejects_invalid_model(self):
        with self.assertRaises(onnx.checker.ValidationError):
            torch_2_5.check_model(_make_model("missing"))
