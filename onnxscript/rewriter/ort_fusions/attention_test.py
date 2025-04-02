# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import unittest

import numpy as np
import packaging.version
import parameterized

import onnxscript
import onnxscript.ir as ir
import onnxscript.rewriter.ort_fusions._core as xformers
from onnxscript import FLOAT, script
from onnxscript import opset18 as op
from onnxscript.ir.passes.common import shape_inference
from onnxscript.rewriter.ort_fusions._test_utils import ORT_VERSION, assert_allclose, ort_run

msft_op = onnxscript.values.Opset("com.microsoft", 1)


class TestAttentionFusion(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batchsize = 2
        self.seqlen = 8
        self.max_seqlen = 32
        self.headsize = 16
        self.num_heads = 10
        self.input_hidden_size = self.headsize * self.num_heads
        self.q_hidden_size = 160
        self.k_hidden_size = 160
        self.v_hidden_size = 160

    def random_inputs(self, with_past=False):
        """Generate random inputs for the model."""
        B = self.batchsize
        S = self.seqlen
        M = self.max_seqlen
        D = self.input_hidden_size
        N = self.num_heads
        H = self.headsize
        D_qkv = self.q_hidden_size + self.k_hidden_size + self.v_hidden_size

        inputs = {
            "input": np.random.rand(B, S, D).astype(np.float32),
            "weight": np.random.rand(D, D_qkv).astype(np.float32),
            "bias": np.random.rand(D_qkv).astype(np.float32),
        }
        if with_past:
            inputs["past"] = np.random.rand(2, B, N, M, H).astype(np.float32)
        return inputs

    def create_model(self, with_past=False):
        """Create a model with or without past inputs."""
        D = self.input_hidden_size
        Dh_qkv = self.q_hidden_size + self.k_hidden_size + self.v_hidden_size

        @script()
        def model_with_mha(input, weight, bias):
            QKV_no_bias = op.MatMul(input, weight)
            QKV = op.Add(QKV_no_bias, bias)

            query_BSDh = op.Slice(QKV, [0], [160], [2])
            key_BSDh = op.Slice(QKV, [160], [320], [2])
            value_BSDh = op.Slice(QKV, [320], [480], [2])

            mha = msft_op.MultiHeadAttention(
                query_BSDh,
                key_BSDh,
                value_BSDh,
                num_heads=self.num_heads,
            )
            return mha

        @script()
        def model_with_mha_past(input, weight, bias, past):
            QKV_no_bias = op.MatMul(input, weight)
            QKV = op.Add(QKV_no_bias, bias)

            query_BSDh = op.Slice(QKV, [0], [160], [2])
            key_BSDh = op.Slice(QKV, [160], [320], [2])
            value_BSDh = op.Slice(QKV, [320], [480], [2])

            past_key_5d = op.Slice(past, [0], [1], [0])
            past_value_5d = op.Slice(past, [1], [2], [0])
            past_key = op.Squeeze(past_key_5d, [0])
            past_value = op.Squeeze(past_value_5d, [0])

            mha, present_key, present_value = msft_op.MultiHeadAttention(
                query_BSDh,
                key_BSDh,
                value_BSDh,
                None,
                None,
                None,
                past_key,
                past_value,
                num_heads=self.num_heads,
            )

            present_key = op.Unsqueeze(present_key, [0])
            present_value = op.Unsqueeze(present_value, [0])
            present = op.Concat(present_key, present_value, axis=0)
            return mha, present

        input_types = (
            FLOAT["B", "S", D],
            FLOAT[D, Dh_qkv],
            FLOAT[Dh_qkv],
        )
        output_types = (FLOAT["B", "S", self.v_hidden_size],)

        if with_past:
            input_types += (FLOAT[2, "B", self.num_heads, "S", self.headsize],)
            output_types += (FLOAT[2, "B", self.num_heads, "NS", self.headsize],)
            model_proto = model_with_mha_past.to_model_proto(
                input_types=input_types,
                output_types=output_types,
            )
        else:
            model_proto = model_with_mha.to_model_proto(
                input_types=input_types,
                output_types=output_types,
            )
        return ir.serde.deserialize_model(model_proto)

    @parameterized.parameterized.expand(
        [
            ("without_past", False),
            ("with_past", True),
        ]
    )
    def test_model_with_mha(self, name, with_past):
        """Test the model with or without past inputs."""
        inputs = self.random_inputs(with_past=with_past)
        model = self.create_model(with_past=with_past)
        model = shape_inference.infer_shapes(model)

        test_with_ort = packaging.version.Version("1.20") <= ORT_VERSION
        if test_with_ort:
            # Run model
            original_outputs = ort_run("original", model, inputs)

        # Fuse Attention
        attention_count = xformers.fuse_attention(model, debug=True)
        self.assertGreater(attention_count, 0)

        if test_with_ort:
            # Run model again
            new_outputs = ort_run("optimized", model, inputs)
            assert_allclose(new_outputs, original_outputs)


if __name__ == "__main__":
    unittest.main()
