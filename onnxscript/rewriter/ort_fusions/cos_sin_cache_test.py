# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

from parameterized import parameterized

import onnxscript.optimizer
from onnxscript.rewriter.ort_fusions._rotary_embedding_models import test_case_1, test_case_2
from onnxscript.rewriter.ort_fusions._smollm_1 import smollm_test_1
from onnxscript.rewriter.ort_fusions._test_utils import assert_allclose, ort_run
from onnxscript.rewriter.ort_fusions.cos_sin_cache import fuse_cos_sin_cache
from onnxscript.rewriter.ort_fusions.rotary_embedding import fuse_rotary_embedding


class TestCosSinCacheTransform(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "smollm_test_1",
                smollm_test_1,
            ),
            (
                "test_case_1",
                test_case_1,
            ),
            (
                "test_case_2",
                test_case_2,
            ),
        ]
    )
    def test_cos_sin_fusion(self, name, test_data_constructor):
        test = test_data_constructor()
        model = test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        inputs = test.get_ort_inputs()
        original_outputs = ort_run("original", model, inputs)
        count = fuse_rotary_embedding(model)
        self.assertGreater(count, 0)
        count = fuse_cos_sin_cache(model)
        self.assertGreater(count, 0)
        new_outputs = ort_run("optimized", model, inputs)
        assert_allclose(new_outputs, original_outputs)


if __name__ == "__main__":
    unittest.main()
