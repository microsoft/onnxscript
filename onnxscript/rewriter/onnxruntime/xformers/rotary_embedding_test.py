# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnxscript.optimizer
from onnxscript.rewriter.onnxruntime.xformers._smollm_1 import TestData
from onnxscript.rewriter.onnxruntime.xformers.rotary_embedding import fuse_rotary_embedding


class TestRotaryEmbedding(unittest.TestCase):
    def test_smollm(self):
        smollm_test = TestData()
        model = smollm_test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        fuse_rotary_embedding(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("RotaryEmbedding", op_types)


if __name__ == "__main__":
    unittest.main()
