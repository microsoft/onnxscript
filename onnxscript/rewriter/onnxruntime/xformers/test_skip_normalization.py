# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

from onnxscript.rewriter.onnxruntime.xformers._test_models import _ort_check, _SmollmTestData
from onnxscript.rewriter.onnxruntime.xformers.skip_normalization import fuse_normalization


class TestSkipNormalization(unittest.TestCase):
    def test_smollm(self):
        smollm_test = _SmollmTestData()
        model = smollm_test.get_onnx_model()
        fuse_normalization(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("SkipSimplifiedLayerNormalization", op_types)
        _ort_check(
            "smollm", model, smollm_test.get_ort_inputs(), smollm_test.get_torch_outputs()
        )


if __name__ == "__main__":
    unittest.main()
