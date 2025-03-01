# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

from parameterized import parameterized

import onnxscript.optimizer
from onnxscript.rewriter.ort_fusions._smollm_1 import TestData as smollm_1_test
from onnxscript.rewriter.ort_fusions._toy_model_1 import TestData as toy_model_1_test
from onnxscript.rewriter.ort_fusions.rotary_embedding import fuse_rotary_embedding


class TestRotaryEmbedding(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "toy_model_1_test",
                toy_model_1_test,
            ),
            (
                "smollm_1_test",
                smollm_1_test,
            ),
        ]
    )
    def test_rotary_embedding_fusion(self, name, test_data_constructor):
        test = test_data_constructor()
        model = test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        fuse_rotary_embedding(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("RotaryEmbedding", op_types)


if __name__ == "__main__":
    unittest.main()
