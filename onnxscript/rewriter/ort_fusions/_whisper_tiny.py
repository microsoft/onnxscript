# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import onnx
from onnx.external_data_helper import (
    convert_model_to_external_data,
    load_external_data_for_model,
)

import onnxscript.ir as ir
import onnxscript.optimizer
import onnxscript.rewriter.ort_fusions._core as xformers


def make_encoder_model():
    pass


def make_decoder_model():
    pass


class TestMultiHeadAttention(unittest.TestCase):
    def test_whisper_tiny(self):
        # Generate encoder model
        whisper_encoder_model = onnx.load(
            "/workspace/testing/whisper-opt/whisper-tiny-4.48/whisper-tiny_encoder.onnx"
        )
        load_external_data_for_model(whisper_encoder_model, "whisper-tiny_encoder.onnx.data")
        encoder_model = ir.serde.deserialize_model(whisper_encoder_model)

        onnxscript.optimizer.optimize(encoder_model)
        encoder_model, fusion_count_e = xformers.fuse_xformers(encoder_model)
        encoder_model.opset_imports["ai.onnxruntime.fusion"] = 1

        print(f"Fused {fusion_count_e} ops")
        self.assertEqual(fusion_count_e["skip_layer_normalization"], 17)
        self.assertEqual(fusion_count_e["sdpa"], 4)
        self.assertEqual(fusion_count_e["mha"], 4)
        self.assertEqual(fusion_count_e["attention"], 4)

        new_encoder_onnx_model = ir.serde.serialize_model(encoder_model)
        convert_model_to_external_data(
            new_encoder_onnx_model,
            all_tensors_to_one_file=True,
            location="whisper-tiny_encoder_optimized.onnx.data",
            size_threshold=1024,
            convert_attribute=False,
        )
        onnx.save(
            new_encoder_onnx_model,
            "/workspace/testing/whisper-opt/whisper-tiny-4.48/whisper-tiny_encoder_optimized.onnx",
        )

        # Generate decoder model
        whisper_decoder_model = onnx.load(
            "/workspace/testing/whisper-opt/whisper-tiny-4.48/whisper-tiny_decoder.onnx"
        )
        load_external_data_for_model(whisper_decoder_model, "whisper-tiny_decoder.onnx.data")
        decoder_model = ir.serde.deserialize_model(whisper_decoder_model)

        onnxscript.optimizer.optimize(decoder_model)
        decoder_model, fusion_count_d = xformers.fuse_xformers(decoder_model)
        decoder_model.opset_imports["ai.onnxruntime.fusion"] = 1

        print(f"Fused {fusion_count_d} ops")
        self.assertEqual(fusion_count_d["skip_layer_normalization"], 25)
        self.assertEqual(fusion_count_d["sdpa"], 8)
        # 4 self-attention + 4 cross-attention
        self.assertEqual(fusion_count_d["mha"], 8)

        new_decoder_onnx_model = ir.serde.serialize_model(decoder_model)
        convert_model_to_external_data(
            new_decoder_onnx_model,
            all_tensors_to_one_file=True,
            location="whisper-tiny_decoder_optimized.onnx.data",
            size_threshold=1024,
            convert_attribute=False,
        )
        onnx.save(
            new_decoder_onnx_model,
            "/workspace/testing/whisper-opt/whisper-tiny-4.48/whisper-tiny_decoder_optimized.onnx",
        )

        """
        test_with_ort = packaging.version.Version("1.20") <= ORT_VERSION
        if test_with_ort:
            # Run model
            inputs = smollm_test.get_ort_inputs()
            original_outputs = ort_run("original", model, inputs)
        """

        # self.assertEqual(fusion_count["sdpa"], )
        # Fuse SDPA and MHA
        # sdpa_count = xformers.fuse_sdpa(model)
        # self.assertGreater(sdpa_count, 0)
        # mha_count = xformers.fuse_mha(model)
        # self.assertGreater(mha_count, 0)

        """
        if test_with_ort:
            # Run model again
            new_outputs = ort_run("optimized", model, inputs)
            assert_allclose(new_outputs, original_outputs)
        """


if __name__ == "__main__":
    unittest.main()
