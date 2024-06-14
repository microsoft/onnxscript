# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pylint: disable=import-outside-toplevel
from __future__ import annotations

import pathlib
import unittest

import onnx
import onnx.backend.test
import parameterized

import onnxscript.testing
from onnxscript import ir

model_folder_path = pathlib.Path(__file__).resolve().parent.parent.parent / "testdata"
onnx_backend_test_path = pathlib.Path(onnx.backend.test.__file__).parent / "data"

model_paths = list(model_folder_path.rglob("*.onnx")) + list(
    onnx_backend_test_path.rglob("*.onnx")
)
test_args = [
    (f"{model_path.parent.name}_{model_path.name}", model_path) for model_path in model_paths
]


class SerdeTest(unittest.TestCase):
    @parameterized.parameterized.expand(test_args)
    def test_serialization_deserialization_produces_same_model(
        self, _: str, model_path: pathlib.Path
    ) -> None:
        model = onnx.load(model_path)
        # Fix the missing graph name of some test models
        model.graph.name = "main_graph"
        onnx.checker.check_model(model)

        # Profile the serialization and deserialization process
        ir_model = ir.serde.deserialize_model(model)
        serialized = ir.serde.serialize_model(ir_model)

        onnxscript.testing.assert_onnx_proto_equal(serialized, model)
        onnx.checker.check_model(serialized)


if __name__ == "__main__":
    unittest.main()
