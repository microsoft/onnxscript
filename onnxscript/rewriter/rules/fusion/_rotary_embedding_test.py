# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx
import onnx_ir as ir
from packaging.version import Version
from parameterized import parameterized

import onnxscript
import onnxscript.rewriter.testing
from onnxscript.rewriter.models import _rotary_embedding_models
from onnxscript.rewriter.rules.fusion import _rotary_embedding


class RotaryEmbeddingOnnxFusionTest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "test_case_1",
                _rotary_embedding_models.test_case_1,
            ),
            (
                "test_case_2",
                _rotary_embedding_models.test_case_2,
            ),
        ]
    )
    def test_rotary_embedding_fusion(self, _: str, test_data_constructor):
        test = test_data_constructor()
        model: ir.Model = test.get_onnx_model()
        model.graph.opset_imports[""] = 23
        model_proto = ir.serde.serialize_model(model)
        onnxscript.optimizer.optimize(model)
        _rotary_embedding.fuse_rotary_embedding(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("RotaryEmbedding", op_types)
        rewritten_model_proto = ir.serde.serialize_model(model)
        inputs = test.get_ort_inputs()

        onnx_version = Version(onnx.__version__)
        min_version = Version("1.19.1")
        is_stable = not (onnx_version.is_devrelease or onnx_version.is_prerelease)
        if onnx_version >= min_version and is_stable:
            onnxscript.rewriter.testing.assert_numerically_equal(
                model_proto, rewritten_model_proto, args=inputs, use_reference=True
            )


if __name__ == "__main__":
    unittest.main()
