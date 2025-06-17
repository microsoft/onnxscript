# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnxscript.optimizer
from onnxscript.rewriter.ort_fusions._test_utils import assert_allclose, ort_run
from onnxscript.rewriter.ort_fusions.models._smollm_1 import smollm_test_1
from onnxscript.rewriter.ort_fusions.models._whisper_decoder import whisper_decoder_test
from onnxscript.rewriter.ort_fusions.models._whisper_encoder import whisper_encoder_test
from onnxscript.rewriter.ort_fusions.rms_normalization import fuse_rms_normalization
from onnxscript.rewriter.ort_fusions.skip_normalization import (
    fuse_skip_layer_normalization,
    fuse_skip_rms_normalization,
)


class TestSkipNormalization(unittest.TestCase):
    def test_smollm(self):
        smollm_test = smollm_test_1()
        model = smollm_test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        inputs = smollm_test.get_ort_inputs()
        original_outputs = ort_run("original", model, inputs)
        fuse_rms_normalization(model)
        fuse_skip_rms_normalization(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("SkipSimplifiedLayerNormalization", op_types)
        new_outputs = ort_run("optimized", model, inputs)
        assert_allclose(new_outputs, original_outputs)

    # TODO: investigate, why precision drops.
    @unittest.skip("fixme: accuracy is not high")
    def test_whisper_encoder(self):
        whisper_encoder = whisper_encoder_test()
        model = whisper_encoder.get_onnx_model()
        onnxscript.optimizer.optimize(model)

        inputs = whisper_encoder.get_ort_inputs()
        original_outputs = ort_run("original", model, inputs)

        fuse_skip_layer_normalization(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("SkipLayerNormalization", op_types)

        new_outputs = ort_run("optimized", model, inputs)
        assert_allclose(new_outputs, original_outputs)

    def test_whisper_decoder(self):
        whisper_decoder = whisper_decoder_test()
        model = whisper_decoder.get_onnx_model()
        onnxscript.optimizer.optimize(model)

        inputs = whisper_decoder.get_ort_inputs()
        original_outputs = ort_run("original", model, inputs)

        fuse_skip_layer_normalization(model)
        op_types = [n.op_type for n in model.graph]
        self.assertIn("SkipLayerNormalization", op_types)

        new_outputs = ort_run("optimized", model, inputs)
        assert_allclose(new_outputs, original_outputs)

    # TODO: add more testcases with default attrs.


if __name__ == "__main__":
    unittest.main()
