# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx

import onnxscript.optimizer
from onnxscript.rewriter.onnxruntime.xformers._smollm_1layer import _SmollmTestData
from onnxscript.rewriter.onnxruntime.xformers._test_utils import assert_allclose, ort_run
from onnxscript.rewriter.onnxruntime.xformers.rms_normalization import fuse_rms_normalization


def model_repr(self):
    return f"Model({self.graph.name})"


onnx.ModelProto.__repr__ = model_repr


class TestRmsNormalization(unittest.TestCase):
    def test_smollm(self):
        smollm_test = _SmollmTestData()
        model = smollm_test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        inputs = smollm_test.get_ort_inputs()
        original_outputs = ort_run("original", model, inputs)
        fuse_rms_normalization(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("SimplifiedLayerNormalization", op_types)
        new_outputs = ort_run("optimized", model, inputs)
        assert_allclose(new_outputs, original_outputs)


if __name__ == "__main__":
    unittest.main()
