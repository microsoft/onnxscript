# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

from parameterized import parameterized

import onnxscript.optimizer
from onnxscript.rewriter.models import _rotary_embedding_models, _smollm_1
from onnxscript.rewriter.ort_fusions import rotary_embedding


class TestRotaryEmbedding(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "test_case_1",
                _rotary_embedding_models.test_case_1,
            ),
            (
                "smollm_test_1",
                _smollm_1.smollm_test_1,
            ),
        ]
    )
    def test_rotary_embedding_fusion(self, _: str, test_data_constructor):
        test = test_data_constructor()
        model = test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        rotary_embedding.fuse_rotary_embedding(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("RotaryEmbedding", op_types)


if __name__ == "__main__":
    unittest.main()
