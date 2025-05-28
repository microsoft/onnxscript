# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import packaging.version

import onnxscript.ir.passes.common as common_passes
import onnxscript.optimizer
import onnxscript.rewriter.ort_fusions._core as xformers
from onnxscript.rewriter.ort_fusions._test_utils import ORT_VERSION, assert_allclose, ort_run
from onnxscript.rewriter.ort_fusions.models._smollm_2 import smollm_test_2
from onnxscript.rewriter.ort_fusions.models._whisper_decoder import whisper_decoder_test
from onnxscript.rewriter.ort_fusions.models._whisper_encoder import whisper_encoder_test


class TestMultiHeadAttention(unittest.TestCase):
    def test_smollm(self):
        # Generate model
        smollm_test = smollm_test_2()
        model = smollm_test.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        xformers.fuse_rms_normalization(model)
        xformers.fuse_skip_rms_normalization(model)
        xformers.fuse_rotary_embedding(model)
        xformers.fuse_cos_sin_cache(model)

        test_with_ort = packaging.version.Version("1.20") <= ORT_VERSION
        if test_with_ort:
            # Run model
            inputs = smollm_test.get_ort_inputs()
            original_outputs = ort_run("original", model, inputs)

        # Fuse SDPA and MHA
        sdpa_count = xformers.fuse_sdpa(model)
        self.assertGreater(sdpa_count, 0)
        mha_count = xformers.fuse_mha1(model)
        mha_count += xformers.fuse_mha2(model)
        self.assertGreater(mha_count, 0)

        if test_with_ort:
            # Run model again
            new_outputs = ort_run("optimized", model, inputs)
            assert_allclose(new_outputs, original_outputs)

    def test_whisper_encoder(self):
        # Generate model
        whisper_encoder = whisper_encoder_test()
        model = whisper_encoder.get_onnx_model()
        onnxscript.optimizer.optimize(model)

        test_with_ort = packaging.version.Version("1.20") <= ORT_VERSION
        if test_with_ort:
            # Run model
            inputs = whisper_encoder.get_ort_inputs()
            original_outputs = ort_run("original", model, inputs)

        # Fuse SDPA and MHA
        sdpa_count = xformers.fuse_sdpa(model)
        self.assertGreater(sdpa_count, 0)
        model = common_passes.ShapeInferencePass()(model).model
        mha_count = xformers.fuse_mha1(model)
        mha_count += xformers.fuse_mha2(model)
        self.assertGreater(mha_count, 0)
        onnxscript.optimizer.optimize(model)

        if test_with_ort:
            # Run model again
            new_outputs = ort_run("optimized", model, inputs)
            assert_allclose(new_outputs, original_outputs)

    def test_whisper_decoder(self):
        # Generate model
        whisper_decoder = whisper_decoder_test()
        model = whisper_decoder.get_onnx_model()
        onnxscript.optimizer.optimize(model)

        test_with_ort = packaging.version.Version("1.20") <= ORT_VERSION
        if test_with_ort:
            # Run model
            inputs = whisper_decoder.get_ort_inputs()
            original_outputs = ort_run("original", model, inputs)

        # Fuse SDPA and MHA
        sdpa_count = xformers.fuse_sdpa(model)
        self.assertGreater(sdpa_count, 0)
        model = common_passes.ShapeInferencePass()(model).model
        mha_count = xformers.fuse_mha1(model)
        mha_count += xformers.fuse_mha2(model)
        self.assertGreater(mha_count, 0)
        onnxscript.optimizer.optimize(model)

        if test_with_ort:
            # Run model again
            new_outputs = ort_run("optimized", model, inputs)
            assert_allclose(new_outputs, original_outputs)


if __name__ == "__main__":
    unittest.main()
