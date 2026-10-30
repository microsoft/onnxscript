# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import inspect
import unittest

import onnx
import pytest

import onnxscript
from onnxscript import FLOAT, onnx_opset, onnx_types
from onnxscript import opset28 as op


class OpsetCoverageTest(unittest.TestCase):
    def test_public_opsets(self):
        for version in range(1, 29):
            with self.subTest(version=version):
                name = f"opset{version}"
                opset = getattr(onnxscript, name)
                self.assertIn(name, onnxscript.__all__)
                self.assertIs(opset, onnx_opset.all_opsets[("", version)])
                self.assertEqual(opset.version, version)
                self.assertEqual(opset.domain, "")

    def test_unreleased_opset_is_not_exposed(self):
        self.assertNotIn(("", 29), onnx_opset.all_opsets)
        self.assertFalse(hasattr(onnxscript, "opset29"))

    def test_recent_schema_signatures(self):
        for schema in onnx.defs.get_all_schemas_with_history():
            if schema.domain or schema.deprecated or not 24 <= schema.since_version <= 28:
                continue
            with self.subTest(op=schema.name, version=schema.since_version):
                opset = getattr(onnxscript, f"opset{schema.since_version}")
                method = getattr(type(opset), schema.name)
                self.assertEqual(
                    method.__qualname__.split(".")[0], f"Opset{schema.since_version}"
                )
                signature = inspect.signature(method)
                self.assertLessEqual(set(schema.attributes), set(signature.parameters))


@pytest.mark.skipif(onnx.defs.onnx_opset_version() < 28, reason="Requires ONNX opset 28.")
class Opset28Test(unittest.TestCase):
    def test_new_operator_and_attribute(self):
        @onnxscript.script()
        def swiglu(a: FLOAT[1, 4, 2, 2], b: FLOAT[1, 4, 2, 2]) -> FLOAT[1, 16, 1, 1]:
            gated = op.SwiGLU(a, b)
            return op.SpaceToDepth(gated, blocksize=2, mode="CRD")

        model = swiglu.to_model_proto()
        onnx.checker.check_model(model, full_check=True)
        self.assertEqual(model.opset_import[0].version, 28)
        self.assertEqual(
            [node.op_type for node in model.graph.node], ["SwiGLU", "SpaceToDepth"]
        )
        attributes = {attr.name: attr for attr in model.graph.node[1].attribute}
        self.assertEqual(attributes["mode"].s, b"CRD")


@pytest.mark.skipif(onnx.defs.onnx_opset_version() < 28, reason="Requires ONNX opset 28.")
@pytest.mark.parametrize("tensor_type", (onnx_types.FLOAT6E2M3, onnx_types.FLOAT6E3M2))
def test_float6_cast_annotations(tensor_type):
    @onnxscript.script()
    def cast(x: FLOAT[2]) -> tensor_type[2]:
        return op.Cast(x, to=tensor_type.dtype)

    model = cast.to_model_proto()
    onnx.checker.check_model(model, full_check=True)
    assert model.graph.output[0].type.tensor_type.elem_type == tensor_type.dtype


if __name__ == "__main__":
    unittest.main()
