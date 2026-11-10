# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import onnx_ir as ir
import onnx_ir.passes.common as common_passes
import packaging.version

import onnxscript.optimizer
import onnxscript.rewriter.ort_fusions._core as xformers
from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.rewriter.models._phi2lm import phi2lm_test
from onnxscript.rewriter.models._smollm_2 import smollm_test_2
from onnxscript.rewriter.models._whisper_decoder import whisper_decoder_test
from onnxscript.rewriter.models._whisper_encoder import whisper_encoder_test
from onnxscript.rewriter.ort_fusions._test_utils import ORT_VERSION, assert_allclose, ort_run


@script()
def _bart_cross_attention(query, key, value):
    query_4d = op.Reshape(query, op.Constant(value_ints=[0, 0, 4, 4]))
    query_transposed = op.Transpose(query_4d, perm=[0, 2, 1, 3])
    shape = op.Shape(query_transposed)
    head_size = op.Slice(
        shape, op.Constant(value_ints=[-1]), op.Constant(value_ints=[9223372036854775807])
    )
    scale = op.Sqrt(op.Div(1.0, op.Sqrt(op.Cast(head_size, to=1))))
    scores = op.MatMul(
        op.Mul(query_transposed, scale),
        op.Mul(op.Transpose(key, perm=[0, 2, 3, 1]), scale),
    )
    attention = op.MatMul(op.Softmax(scores, axis=-1), value)
    output = op.Reshape(
        op.Transpose(attention, perm=[0, 2, 1, 3]),
        op.Constant(value_ints=[0, 0, 16]),
    )
    present_key = op.Transpose(key, perm=[0, 2, 1, 3])
    return output, present_key, value


@script()
def _bart_self_attention_with_past(query, key, value, past_key, past_value):
    head_shape = op.Constant(value_ints=[0, 0, 4, 4])
    query_4d = op.Transpose(op.Reshape(query, head_shape), perm=[0, 2, 1, 3])
    key_4d = op.Transpose(op.Reshape(key, head_shape), perm=[0, 2, 1, 3])
    value_4d = op.Transpose(op.Reshape(value, head_shape), perm=[0, 2, 1, 3])
    present_key = op.Concat(past_key, key_4d, axis=-2)
    present_value = op.Concat(past_value, value_4d, axis=-2)
    head_size = op.Slice(
        op.Shape(query_4d),
        op.Constant(value_ints=[-1]),
        op.Constant(value_ints=[9223372036854775807]),
    )
    scale = op.Sqrt(op.Div(1.0, op.Sqrt(op.Cast(head_size, to=1))))
    scores = op.MatMul(
        op.Mul(query_4d, scale),
        op.Mul(op.Transpose(present_key, perm=[0, 1, 3, 2]), scale),
    )
    attention = op.MatMul(op.Softmax(scores, axis=-1), present_value)
    output = op.Reshape(
        op.Transpose(attention, perm=[0, 2, 1, 3]),
        op.Constant(value_ints=[0, 0, 16]),
    )
    return output, present_key, present_value


class TestMultiHeadAttention(unittest.TestCase):
    def test_bart_self_attention_with_past(self):
        model = ir.serde.deserialize_model(
            _bart_self_attention_with_past.to_model_proto(
                input_types=[
                    FLOAT["B", "S", 16],
                    FLOAT["B", "S", 16],
                    FLOAT["B", "S", 16],
                    FLOAT["B", 4, "Spast", 4],
                    FLOAT["B", 4, "Spast", 4],
                ],
                output_types=[
                    FLOAT["B", "S", 16],
                    FLOAT["B", 4, "St", 4],
                    FLOAT["B", 4, "St", 4],
                ],
            )
        )
        rng = np.random.default_rng(1)
        inputs = {
            name: rng.standard_normal((2, 1, 16)).astype(np.float32)
            for name in ("query", "key", "value")
        }
        inputs.update(
            {
                name: rng.standard_normal((2, 4, 5, 4)).astype(np.float32)
                for name in ("past_key", "past_value")
            }
        )
        expected = ort_run("original", model, inputs)
        model, counts = xformers.optimize_for_ort(model)
        self.assertEqual(counts["sdpa"], 1)
        self.assertEqual(counts["mha1"], 1)
        self.assertEqual(counts["sdpa_via_mha"], 0)
        self.assertFalse(any(node.domain == "ai.onnxruntime._fusion" for node in model.graph))
        assert_allclose(ort_run("optimized", model, inputs), expected, rtol=1e-5, atol=1e-6)

    def test_bart_cross_attention(self):
        for key_format in ("BSHd", "BHSd"):
            for sequence_length in (4, 7):
                for full_pipeline in (False, True):
                    with self.subTest(
                        key_format=key_format,
                        sequence_length=sequence_length,
                        full_pipeline=full_pipeline,
                    ):
                        model = ir.serde.deserialize_model(
                            _bart_cross_attention.to_model_proto(
                                input_types=[
                                    FLOAT["B", "S", 16],
                                    FLOAT["B", "Skv", 4, 4],
                                    FLOAT["B", 4, "Skv", 4],
                                ],
                                output_types=[
                                    FLOAT["B", "S", 16],
                                    FLOAT["B", 4, "Skv", 4],
                                    FLOAT["B", 4, "Skv", 4],
                                ],
                            )
                        )
                        kv_length = 4 if sequence_length == 4 else "Skv"
                        model.graph.inputs[1].shape = ir.Shape(["B", kv_length, 4, 4])
                        model.graph.inputs[2].shape = ir.Shape(["B", 4, kv_length, 4])
                        if key_format == "BHSd":
                            model.graph.inputs[1].shape = ir.Shape(["B", 4, kv_length, 4])
                            for node in model.graph:
                                if (
                                    node.op_type == "Transpose"
                                    and node.inputs[0].name == "key"
                                ):
                                    perm = node.attributes.get_ints("perm")
                                    node.attributes["perm"] = ir.AttrInt64s(
                                        "perm",
                                        [0, 1, 3, 2] if perm == (0, 2, 3, 1) else [0, 1, 2, 3],
                                    )
                        rng = np.random.default_rng(0)
                        key = rng.standard_normal((2, sequence_length, 4, 4)).astype(
                            np.float32
                        )
                        if key_format == "BHSd":
                            key = key.transpose(0, 2, 1, 3)
                        inputs = {
                            "query": rng.standard_normal((2, 3, 16)).astype(np.float32),
                            "key": key,
                            "value": rng.standard_normal((2, 4, sequence_length, 4)).astype(
                                np.float32
                            ),
                        }
                        expected = ort_run("original", model, inputs)
                        if full_pipeline:
                            model, counts = xformers.optimize_for_ort(model)
                            self.assertEqual(counts["sdpa"], 1)
                            self.assertEqual(counts["mha2"], 1)
                            self.assertEqual(counts["sdpa_via_mha"], 0)
                        else:
                            onnxscript.optimizer.optimize(model)
                            self.assertEqual(xformers.fuse_sdpa(model), 1)
                            model = common_passes.ShapeInferencePass()(model).model
                            self.assertEqual(xformers.fuse_mha2(model), 1)
                            onnxscript.optimizer.optimize(model)
                        self.assertFalse(
                            any(
                                node.domain == "ai.onnxruntime._fusion" for node in model.graph
                            )
                        )
                        assert_allclose(
                            ort_run("optimized", model, inputs), expected, rtol=1e-5, atol=1e-6
                        )

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
        sdpa_count = xformers.fuse_sdpa(model, debug=True)
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

    def test_phi2lm(self):
        test_case = phi2lm_test()
        model = test_case.get_onnx_model()
        onnxscript.optimizer.optimize(model)
        xformers.optimize_for_ort(model)
        mha_nodes = [n for n in model.graph if n.op_type == "MultiHeadAttention"]
        self.assertEqual(
            len(mha_nodes),
            1,
            "Expected exactly one MultiHeadAttention node after optimization",
        )
        mha_node = mha_nodes[0]
        # Check that the MHA node has past kv cache inputs
        self.assertEqual(len(mha_node.inputs), 8, "Expected MHA node to have 8 inputs")
        self.assertIsNotNone(mha_node.inputs[6], "Expected MHA node to have past key input")
        self.assertIsNotNone(mha_node.inputs[7], "Expected MHA node to have past value input")


if __name__ == "__main__":
    unittest.main()
