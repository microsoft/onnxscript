# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx_ir as ir
from parameterized import parameterized

import onnxscript
from onnxscript.rewriter import onnx_fusions
from onnxscript.rewriter.models import _rotary_embedding_models
import onnxscript.rewriter.testing

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
        for opset_version in [22, 23]:
            model: ir.Model = test.get_onnx_model()
            model.graph.opset_imports[""] = opset_version
            model_proto = ir.serde.serialize_model(model)
            onnxscript.optimizer.optimize(model)
            onnx_fusions.fuse(model)
            op_types = [n.op_type for n in model.graph]
            if opset_version == 22:
                self.assertNotIn("RotaryEmbedding", op_types)
            else:
                self.assertIn("RotaryEmbedding", op_types)
                rewritten_model_proto = ir.serde.serialize_model(model)
                inputs = test.get_ort_inputs()
                onnxscript.rewriter.testing.assert_numerically_equal(
                    model_proto, rewritten_model_proto, args=inputs, use_reference=True
                )


if __name__ == "__main__":
    unittest.main()
