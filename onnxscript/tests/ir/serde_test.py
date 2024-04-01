from __future__ import annotations

import pathlib
import unittest

import onnx
import parameterized

import onnxrewriter.testing
from onnxrewriter.experimental import intermediate_representation as ir

model_folder_path = pathlib.Path(__file__).resolve().parent.parent.parent / "testdata"

model_paths = list(model_folder_path.rglob("*.onnx"))
test_args = [(model_path.name, model_path) for model_path in model_paths]


class SerdeTest(unittest.TestCase):
    @parameterized.parameterized.expand(test_args)
    def test_serialization_deserialization_produces_same_model(
        self, _: str, model_path: pathlib.Path
    ) -> None:
        model = onnx.load(model_path)
        ir_model = ir.serde.deserialize_model(model)
        serialized = ir.serde.serialize_model(ir_model)
        onnxrewriter.testing.assert_onnx_proto_equal(serialized, model)


if __name__ == "__main__":
    unittest.main()
